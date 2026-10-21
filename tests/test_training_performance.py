"""训练提速回归：指标不变、同键只算一次、并发预算、响应重放与诊断。"""
from __future__ import annotations

import copy
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from multi_agent_snd import MultiAgentSNDSolver, UtilityConfig
from multi_agent_snd.baselines import BASE_SOLVER, OfflineMultiAgentClient
from multi_agent_snd.evaluation import SNDEvaluator, digest, lower_bound
from multi_agent_snd.performance import ordered_map, TimingRecorder
from multi_agent_snd.replay import ReplayClient
from multi_agent_snd.storage import ProgramArchive
from snd_generator import SNDInstance


def legacy_bound(data, field):
    waiting = {str(a['arc_id']) for a in data.get('time_expansion', {}).get('arc_schedule', [])
               if a['kind'] == 'wait'}
    total = 0.0
    for k in data['commodities']:
        distances = dict.fromkeys(data['nodes'], math.inf)
        distances[k['origin']] = 0.0
        for _ in range(len(distances) - 1):
            for a in data['arcs']:
                weight = float(a[field]) if field else (0.0 if str(a['id']) in waiting else 1.0)
                distances[a['head']] = min(distances[a['head']], distances[a['tail']] + weight)
        total += distances[k['destination']] * float(k['demand'])
    return total


class TestTrainingPerformance(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = SNDInstance(4, .7, .5, seed=7).to_dict()
        self.archive = ProgramArchive(self.root)
        self.evaluator = SNDEvaluator(self.data, self.archive, UtilityConfig(),
                                      'scipy', 10, 0, 10, 1, 42, parallel_workers=2)

    def program(self, code=BASE_SOLVER):
        return self.archive.add(player='solver', role='solver', explanation='test', code=code,
                                parents=[], iteration=1, eoh_iteration=0, operator='I1')

    def test_optimized_bounds_match_old_algorithm_static_and_temporal(self):
        for temporal in (False, True):
            for seed in range(4):
                data = SNDInstance(5, .7, .5, seed=seed,
                                   time_periods=3 if temporal else None).to_dict()
                for field in (None, 'variable_cost'):
                    with self.subTest(temporal=temporal, seed=seed, field=field):
                        self.assertAlmostEqual(lower_bound(data, field), legacy_bound(data, field), places=8)

    def test_bounds_parallel_zero_edges_and_unreachable(self):
        data = dict(nodes=['a', 'b', 'c'], arcs=[
            dict(id=0, tail='a', head='b', variable_cost=0),
            dict(id=1, tail='a', head='b', variable_cost=4),
            dict(id=2, tail='b', head='c', variable_cost=2)], commodities=[
                dict(origin='a', destination='c', demand=3),
                dict(origin='a', destination='b', demand=2)])
        self.assertEqual(lower_bound(data, None), 8)
        self.assertEqual(lower_bound(data, 'variable_cost'), 6)
        data['arcs'] = []
        self.assertEqual(lower_bound(data, None), math.inf)

    def test_dijkstra_rejects_negative_weight(self):
        data = copy.deepcopy(self.data)
        data['arcs'][0]['variable_cost'] = -1
        with self.assertRaises(ValueError):
            lower_bound(data, 'variable_cost')

    def test_lower_bound_cache_is_content_addressed_and_single_flight(self):
        original = lower_bound
        def slow(*args):
            time.sleep(.02)
            return original(*args)
        with patch('multi_agent_snd.evaluation.lower_bound', side_effect=slow) as compute:
            with ThreadPoolExecutor(max_workers=6) as pool:
                rows = list(pool.map(self.evaluator.instance_lower_bounds,
                                     [copy.deepcopy(self.data) for _ in range(6)]))
            self.assertEqual(compute.call_count, 2)
            self.assertTrue(all(row == rows[0] for row in rows))
            for field in ('capacity', 'variable_cost'):
                data = copy.deepcopy(self.data)
                data['arcs'][0][field] *= 1.2
                self.evaluator.instance_lower_bounds(data)
            data = copy.deepcopy(self.data)
            data['commodities'][0]['demand'] *= 1.2
            self.evaluator.instance_lower_bounds(data)
            self.assertEqual(compute.call_count, 8)
        self.assertEqual(len(self.evaluator.lower_bounds), 4)
        self.assertGreater(self.evaluator.timing.summary()['counters']['lower_bounds_cache_wait'], 0)

    def test_failed_single_flight_releases_waiters_and_allows_retry(self):
        barrier = threading.Barrier(3)
        def fail(_):
            barrier.wait(timeout=3)
            return self.evaluator.instance_lower_bounds(self.data)
        def bad(*args):
            time.sleep(.05)
            raise ValueError('bound failed')
        with patch('multi_agent_snd.evaluation.lower_bound', side_effect=bad) as compute:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(fail, i) for i in range(3)]
                for future in futures:
                    with self.assertRaisesRegex(ValueError, 'bound failed'):
                        future.result(timeout=3)
            self.assertEqual(compute.call_count, 1)
        self.assertTrue(self.evaluator.instance_lower_bounds(self.data))

    def test_different_candidates_reuse_bounds_and_same_candidate_runs_once(self):
        data = dict(nodes=[1, 2], arcs=[dict(id=0, tail=1, head=2, capacity=10,
                                          fixed_cost=2, variable_cost=3)],
                    commodities=[dict(id=0, origin=1, destination=2, demand=4)])
        solution = dict(arc_open={'0': 1}, flows={'0': {'0': 4}})
        first, second = self.program(), self.program(BASE_SOLVER + '\n# variant')
        with patch.object(self.evaluator, 'reference', return_value={'value': 14, 'kind': 'heuristic'}), \
                patch('multi_agent_snd.evaluation.execute', return_value={
                    'result': solution, 'lines': 5, 'elapsed': .01, 'wall_time': .02}) as execute, \
                patch('multi_agent_snd.evaluation.lower_bound', wraps=lower_bound) as bound:
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(lambda _: self.evaluator.evaluate(first, data), range(4)))
            other = self.evaluator.evaluate(second, data)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(bound.call_count, 2)
        self.assertEqual(rows[0]['metrics'], other['metrics'])

    def test_reference_and_sample_single_flight(self):
        generator = self.archive.add(player='cost', role='generator', explanation='test',
            code='def generate_snd(base_instance, seed):\n    return dict(base_instance)',
            parents=[], iteration=1, eoh_iteration=0, operator='I1')
        def reference(data):
            time.sleep(.02)
            return {'kind': 'heuristic', 'value': 1}
        with patch.object(self.evaluator, '_reference', side_effect=reference) as compute:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(self.evaluator.reference, [self.data] * 4))
            self.assertEqual(compute.call_count, 1)
        with patch('multi_agent_snd.evaluation.execute', return_value={'result': self.data}) as execute:
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(lambda _: self.evaluator.sample(generator, 0), range(4)))
            self.assertTrue(all(row['status'] == 'ok' for row in rows))
            self.assertEqual(execute.call_count, 1)

    def test_nested_maps_do_not_create_nested_pools(self):
        def inner(_):
            outer_thread = threading.get_ident()
            return ordered_map(lambda _: threading.get_ident() == outer_thread, range(3), 3)
        self.assertEqual(ordered_map(inner, range(3), 2), [[True]*3]*3)

    def test_candidate_process_budget_is_global(self):
        lock = threading.Lock()
        active = peak = 0
        def execute(*args, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.02)
            with lock:
                active -= 1
        with patch('multi_agent_snd.evaluation.execute', side_effect=execute):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: self.evaluator._execute('', 'solve_snd', {}), range(8)))
        self.assertEqual(peak, 2)

    def test_timing_records_failure_and_valid_jsonl(self):
        with self.assertRaises(ValueError):
            with self.evaluator.timing.measure('intentional'):
                raise ValueError('test')
        stats = self.evaluator.timing.summary()['stages']['intentional']
        self.assertEqual(stats['failures'], 1)
        self.assertEqual(json.loads((self.root / 'timings.jsonl').read_text())['stage'], 'intentional')

    def test_optional_config_validation(self):
        for kwargs in ({'response_workers': 0}, {'generator_operators': ()},
                       {'generator_operators': ('M1', 'M1')}, {'generator_operators': ('BAD',)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                MultiAgentSNDSolver(self.data, **kwargs)

    def test_parallel_responses_share_old_snapshot_and_stable_seeds(self):
        def run(reverse=False):
            solver = MultiAgentSNDSolver(self.data, response_workers=2, progress_output=False,
                                          generator_directions=('cost', 'path', 'congestion'), make_plots=False)
            root = self.root / ('reversed' if reverse else 'normal')
            root.mkdir()
            solver.archive = ProgramArchive(root)
            solver.timing = TimingRecorder(solver.archive)
            solver.user_prompt = 'test'
            pools = {player: [] for player in solver.players}
            meta = {'marginals': [[] for _ in solver.players]}
            def search(player, t, actual_pools, actual_meta):
                self.assertIs(actual_pools, pools)
                self.assertIs(actual_meta, meta)
                time.sleep(.02 if (player == 'solver') != reverse else 0)
                return solver._generate(player, 'I1', [], t, 0, pools, meta)[0]
            def llm(kind, arguments, timeout):
                client, prompt, temperature, n, seed = arguments
                return {'result': OfflineMultiAgentClient().generate(prompt, temperature, n)}
            with patch.object(solver, '_search', side_effect=search), \
                    patch('multi_agent_snd.solver.isolated_call', side_effect=llm):
                response = solver._search_responses(1, pools, meta)
            calls = {call['player']: (call['seed'], call['prompt']) for call in solver.archive.calls}
            self.assertEqual(len(calls), 4)
            self.assertEqual(len({seed for seed, _ in calls.values()}), 4)
            self.assertEqual(sorted(call['index'] for call in solver.archive.calls), list(range(4)))
            return response, calls
        first, first_calls = run()
        second, second_calls = run(reverse=True)
        self.assertEqual(first_calls, second_calls)
        self.assertEqual({k: v.id for k,v in first.items()}, {k: v.id for k,v in second.items()})
        self.assertNotEqual(first_calls['solver'][0], first_calls['cost'][0])

    def test_generator_operator_budget_does_not_change_solver_operators(self):
        solver = MultiAgentSNDSolver(self.data, generator_directions=('cost',),
                                     generator_operators=('M1',), population_size=2,
                                     eoh_iterations=1, progress_output=False, make_plots=False)
        solver.archive = self.archive
        calls = []
        def generate(player, operator, *args):
            calls.append((player, operator))
            return [self.archive.add(player=player, role=player, explanation='test',
                code=f'def test():\n    return {len(calls)}', parents=[], iteration=1,
                eoh_iteration=0, operator=operator)]
        pools = {'solver': [self.program()], 'cost': []}
        with patch.object(solver, '_generate', side_effect=generate), \
                patch.object(solver, '_fitness', return_value=1.0):
            solver._search_eoh_original('cost', 1, pools, {})
            solver._search_eoh_original('solver', 1, pools, {})
        self.assertEqual([op for player, op in calls if player == 'cost'], ['I1', 'M1'])
        self.assertEqual([op for player, op in calls if player == 'solver'],
                         ['I1', 'E1', 'E2', 'M1', 'M2', 'M3'])

    def test_parallel_archive_has_unique_ids_and_complete_jsonl(self):
        def write(player):
            with self.archive.scope(f't001_{player}'):
                for index in range(10):
                    self.archive.add(player=player, role='generator', explanation='test',
                        code=f'def test():\n    return {index}', parents=[], iteration=1,
                        eoh_iteration=0, operator='M1')
        ordered_map(write, ('cost', 'path', 'congestion'), 3)
        rows = [json.loads(line) for line in (self.root / 'programs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 30)
        self.assertEqual(len({row['id'] for row in rows}), 30)

    def test_parallel_offline_integration_and_replay(self):
        kwargs = dict(generator_directions=('cost', 'path', 'congestion'), outer_iterations=1, eoh_iterations=1,
            population_size=2, n_candidates=1, instances_per_generator=1,
            eoh_framework='eoh-s', response_mode='mpsro', response_workers=2,
            generator_operators=('M1',), parallel_workers=2, progress_output=False,
            make_plots=False, test_instance_count=1, validation_instance_count=1,
            reference_method='heuristic', reference_restarts=1)
        first = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7),
                                    output_dir=self.root / 'first', **kwargs).solve('test parallel')
        replay = ReplayClient(first.output_dir / 'llm_calls.jsonl')
        second = MultiAgentSNDSolver(SNDInstance(3, .7, .5, seed=7), llm_client=replay,
                                     output_dir=self.root / 'second', **kwargs).solve('test parallel')
        self.assertEqual(first.data['objective'], second.data['objective'])
        self.assertEqual(first.data['selected_solver']['code_hash'], second.data['selected_solver']['code_hash'])
        self.assertTrue((first.output_dir / 'timing_summary.json').is_file())
        self.assertEqual(first.data['llm_summary']['successful_calls'], second.data['llm_summary']['successful_calls'])
        selected = first.data['selected_solver']
        self.assertNotEqual(selected['operator'], 'base')
        self.assertEqual(first.data['config']['response_workers'], 2)
        self.assertEqual(first.data['config']['generator_operators'], ['M1'])
