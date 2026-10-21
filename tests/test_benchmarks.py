import unittest
from pathlib import Path
import io
import json
import shutil
from copy import deepcopy
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import patch
import zipfile

from multi_agent_snd.benchmarks import (load_canad_benchmark, load_holmberg_benchmark,
                                       load_fnss_benchmark, load_mnetgen_benchmark, load_benchmark, CANAD_URLS,
                                       capacity_infeasibility_certificates)
from snd_solver import SNDSolver


class TestCanadBenchmark(unittest.TestCase):
    def test_published_r_instances_preserve_data(self):
        archive = Path(__file__).resolve().parents[1] / 'benchmarks' / 'R.tgz'
        if not archive.exists():
            self.skipTest('Download public benchmark archive first')
        for name, commodities in [('r01.1.dow', 10), ('r02.1.dow', 25), ('r03.1.dow', 50)]:
            data = load_canad_benchmark(archive, name)
            self.assertEqual(len(data['nodes']), 10)
            self.assertEqual(len(data['arcs']), 35)
            self.assertEqual(len(data['commodities']), commodities)
            self.assertEqual(data['arcs'][0]['variable_cost'], 100)
            self.assertEqual(len(data['metadata']['source_sha256']), 64)
            nodes, arcs, demands = SNDSolver._normalize(data)
            self.assertEqual(len(nodes), 10)
            self.assertEqual(len(arcs), 35)
            self.assertEqual(len(demands), commodities)

    def test_missing_member_is_not_silently_generated(self):
        archive = Path(__file__).resolve().parents[1] / 'benchmarks' / 'R.tgz'
        if not archive.exists():
            self.skipTest('Download public benchmark archive first')
        with self.assertRaises(KeyError):
            load_canad_benchmark(archive, 'missing.dow')


class TestBenchmarkAdapters(unittest.TestCase):
    @staticmethod
    def _write_tar(path, members):
        with tarfile.open(path, 'w:gz') as bundle:
            for name, content in members.items():
                raw = content.encode('ascii')
                member = tarfile.TarInfo(name)
                member.size = len(raw)
                bundle.addfile(member, io.BytesIO(raw))

    def test_canad_family_provenance(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'CPlus.tgz'
            self._write_tar(path, {'small.dow': '2 1 1\n1 2 3 10 7 1 1\n1 2 4\n'})
            data = load_canad_benchmark(path, 'small.dow')
            self.assertEqual(data['metadata']['source_url'], CANAD_URLS['CPlus'])
            self.assertEqual(data['arcs'][0]['fixed_cost'], 7)
            self.assertEqual(data['commodities'][0]['demand'], 4)

    def test_holmberg_free_and_paid_capacity_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'test.zip'
            with zipfile.ZipFile(path, 'w') as bundle:
                bundle.writestr('00index1', 'cex1/test:3-2-1-1\n')
                bundle.writestr('test', '1 2 3 5 0\n2 3 4 0 1\n7 10\n'
                                '0 0 0 0 0\n9 1 3 4\n0 0 0 0\n')
            data = load_holmberg_benchmark(path, 'test')
            self.assertEqual(len(data['nodes']), 3)
            self.assertEqual([a['capacity'] for a in data['arcs']], [5, 10])
            self.assertEqual([a['fixed_cost'] for a in data['arcs']], [0, 7])
            self.assertEqual(data['commodities'][0]['id'], 9)
            self.assertEqual(data['metadata']['comparison_group'], 'fixed_charge_snd')

    def test_holmberg_unsupported_capacity_is_not_silently_merged(self):
        for link in ('1 2 3 1 1', '1 2 3 0 2'):
            with self.subTest(link=link), TemporaryDirectory() as directory:
                path = Path(directory) / 'test.zip'
                with zipfile.ZipFile(path, 'w') as bundle:
                    bundle.writestr('00index1', 'cex1/test:2-1-1\n')
                    bundle.writestr('test', link + '\n7 10\n0 0 0 0 0\n1 1 2 4\n0 0 0 0\n')
                with self.assertRaises(ValueError):
                    load_holmberg_benchmark(path, 'test')

    def _fnss_files(self, arc='1 1 2 -1 3 10 1', supply='1 1 4\n2 1 -4'):
        return {'waxman/test.nod': '1 2 1 1', 'waxman/test.arc': arc,
                'waxman/test.mut': '1 10', 'waxman/test.sup': supply}

    def test_fnss_zero_fixed_cost_and_supply_sign_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'test.tgz'
            self._write_tar(path, self._fnss_files())
            data = load_fnss_benchmark(path, 'test')
            self.assertEqual(data['arcs'][0]['fixed_cost'], 0)
            self.assertEqual(data['arcs'][0]['capacity'], 10)
            self.assertEqual(data['arcs'][0]['variable_cost'], 3)
            self.assertEqual(data['commodities'][0], dict(id=1, origin=1, destination=2, demand=4))
            self.assertEqual(data['metadata']['comparison_group'], 'linear_flow_special_case')
            self.assertEqual(len(data['metadata']['source_files_sha256']), 4)

    def test_fnss_does_not_drop_unsupported_constraints(self):
        for arc, supply in [('1 1 2 1 3 10 1', '1 1 4\n2 1 -4'),
                            ('1 1 2 -1 3 5 1', '1 1 4\n2 1 -4'),
                            ('1 1 2 -1 3 10 1', '1 1 4\n2 1 -3')]:
            with self.subTest(arc=arc, supply=supply), TemporaryDirectory() as directory:
                path = Path(directory) / 'test.tgz'
                self._write_tar(path, self._fnss_files(arc, supply))
                with self.assertRaises(ValueError):
                    load_fnss_benchmark(path, 'test')

    def test_mnetgen_reader_supports_planar_style_prefix(self):
        for prefix in ('planarmnet', 'gridmnet'):
            with self.subTest(prefix=prefix), TemporaryDirectory() as directory:
                path = Path(directory) / f'{prefix}.tgz'
                self._write_tar(path, {
                    f'{prefix}/sample.nod': '2 3 3 3',
                    f'{prefix}/sample.arc': '1 1 2 -1 3 10 1\n2 2 3 -1 4 10 2\n3 1 3 -1 9 10 3',
                    f'{prefix}/sample.mut': '1 10\n2 10\n3 10',
                    f'{prefix}/sample.sup': '1 1 5\n3 1 -5\n1 2 4\n3 2 -4',
                })
                data = load_mnetgen_benchmark(path, 'sample')
                self.assertEqual(len(data['nodes']), 3)
                self.assertEqual(len(data['arcs']), 3)
                self.assertEqual(data['metadata']['comparison_group'], 'linear_flow_special_case')

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            load_benchmark('missing.zip', 'anything', kind='unknown')

    def test_selected_public_suite_covers_10_to_200_nodes(self):
        import final_code_test as demo
        self.assertEqual(len(demo.BENCHMARK_CASES), 32)
        self.assertEqual(len(set(demo.BENCHMARK_CASES)), 32)
        self.assertFalse(any(name in ('planar300', 'planar500') for _, _, name in demo.BENCHMARK_CASES))
        expected_counts = {'r01.1.dow': (10, 35, 10), 'cexc4': (50, 290, 40),
                           'cexf6': (150, 410, 10), 'w1_200_04': (200, 1550, 1528)}
        if not all((demo.BENCHMARK_DIR / archive).exists() for _, archive, _ in demo.BENCHMARK_CASES):
            self.skipTest('Download public benchmark archives first')
        sizes, hashes = set(), []
        for kind, archive, name in demo.BENCHMARK_CASES:
            data = load_benchmark(demo.BENCHMARK_DIR / archive, name, kind=kind)
            counts = (len(data['nodes']), len(data['arcs']), len(data['commodities']))
            sizes.add(counts[0])
            hashes.append(data['metadata']['source_sha256'])
            if name in expected_counts:
                self.assertEqual(counts, expected_counts[name])
        self.assertEqual(min(sizes), 10)
        self.assertEqual(max(sizes), 200)
        self.assertTrue({10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 100, 103, 150, 200} <= sizes)
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_historical_solver_comparison_end_to_end(self):
        import final_code_test as demo
        archive = demo.BENCHMARK_DIR / 'R.tgz'
        if not archive.exists() or not demo.SOLVER_FILE.exists():
            self.skipTest('Historical exported solver and public archive required')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile(archive, root / 'R.tgz')
            with patch.multiple(demo,
                    BENCHMARK_CASES=(('canad', 'R.tgz', 'r01.1.dow'),),
                    BENCHMARK_DIR=root, CONVERTED_INSTANCE_DIR=root / 'converted',
                    OUTPUT_FILE=root / 'final_solution.json',
                    COMPARISON_FILE=root / 'final_comparison.json',
                    FIGURE_OUTPUT_DIR=root / 'figures', FIGURE_DPI=80,
                    SOLVER_TIMEOUT=15.0, SOLVER_LINE_LIMIT=30_000_000,
                    BASELINE_TIMEOUT=5.0, GRASP_RESTARTS=2,
                    PLOT_ONLY=False, GENERATE_FIGURES=True):
                demo.main()
                report = json.loads(demo.COMPARISON_FILE.read_text(encoding='utf-8'))
                self.assertEqual(report['status'], 'completed')
                self.assertEqual(report['completed_instances'], 1)
                self.assertNotIn('visualization_error', report)
                rows = report['instances'][0]['results']
                self.assertEqual(len(rows), 5)
                self.assertTrue(all(row['status'] == 'ok' for row in rows), rows)
                self.assertEqual(rows[0]['method'], 'exported_llm_heuristic')
                self.assertTrue(Path(report['figure_index']).exists())
                self.assertEqual(len(report['figure_files']), 4)
                self.assertTrue((Path(report['result_dir']) / 'comparison.json').exists())
                first_dir = Path(report['figure_output_dir'])
                # 绘图模式只读现有报告，不重新调用求解器或覆盖旧图。
                with patch.object(demo, 'PLOT_ONLY', True), \
                        patch.object(demo, 'load_solver_class', side_effect=AssertionError('unexpected solve')):
                    demo.main()
                replotted = json.loads(demo.COMPARISON_FILE.read_text(encoding='utf-8'))
                self.assertNotEqual(first_dir, Path(replotted['figure_output_dir']))
                self.assertTrue(first_dir.exists())
                self.assertEqual(report['instances'], replotted['instances'])

    def test_large_public_linear_suite_is_discoverable(self):
        import final_code_test as demo
        cases = [case for case in demo.BENCHMARK_CASES if case[0] == 'mnetgen']
        self.assertTrue(cases)
        if not all((demo.BENCHMARK_DIR / archive).exists() for _, archive, _ in cases):
            self.skipTest('Download large Mnetgen benchmark archives first')
        sizes = []
        for kind, archive, name in cases:
            data = load_benchmark(demo.BENCHMARK_DIR / archive, name, kind=kind)
            sizes.append(len(data['nodes']))
            self.assertEqual(data['metadata']['comparison_group'], 'linear_flow_special_case')
            self.assertTrue(all(float(a['fixed_cost']) == 0 for a in data['arcs']))
        self.assertEqual(set(sizes), {25, 30, 50, 80, 100, 150})
        self.assertTrue(all(size <= 200 for size in sizes))

    def test_baseline_deadline_is_enforced(self):
        from final_code_test import _baseline_solution
        data = dict(nodes=[1, 2], arcs=[dict(id=0, tail=1, head=2, variable_cost=1,
                                            fixed_cost=2, capacity=10)],
                    commodities=[dict(id=0, origin=1, destination=2, demand=3)])
        with self.assertRaises(TimeoutError):
            _baseline_solution(data, 'variable_cost', deadline=0)

    def test_singleton_capacity_cut_certifies_impossible_input(self):
        data = dict(nodes=[1, 2], arcs=[dict(tail=1, head=2, capacity=5)],
                    commodities=[dict(origin=1, destination=2, demand=7)])
        checks = capacity_infeasibility_certificates(data)
        self.assertEqual(len(checks), 2)
        self.assertEqual(checks[0]['deficit'], 2)
        data['commodities'][0]['demand'] = 4
        self.assertEqual(capacity_infeasibility_certificates(data), [])

    def test_skipped_infeasible_input_not_counted_as_algorithm_failure(self):
        from final_code_test import summarize_comparison
        case = dict(nodes=200, metadata=dict(comparison_group='linear_flow_special_case'),
                    input_audit=dict(status='proven_infeasible'),
                    results=[dict(method='test', status='skipped_infeasible_input',
                                  feasible=None, solve_time=0, best_observed_relative_gap=None)])
        group = summarize_comparison([case])['linear_flow_special_case']
        self.assertEqual(group['proven_infeasible_inputs'], 1)
        result = group['methods']['test']
        self.assertEqual(result['attempted_instances'], 0)
        self.assertEqual(result['failed_instances'], 0)
        self.assertIsNone(result['success_rate'])

    def test_summary_separates_models_and_counts_failures(self):
        from final_code_test import summarize_comparison
        success = dict(method='test', status='ok', feasible=True, solve_time=1,
                       best_observed_relative_gap=0.2)
        failure = dict(method='test', status='failed', feasible=False, solve_time=2,
                       best_observed_relative_gap=None)
        cases = [dict(metadata=dict(comparison_group='fixed_charge_snd'), nodes=10, results=[success]),
                 dict(metadata=dict(comparison_group='fixed_charge_snd'), nodes=20, results=[failure]),
                 dict(metadata=dict(comparison_group='linear_flow_special_case'), nodes=200, results=[success])]
        summary = summarize_comparison(cases)
        fixed = summary['fixed_charge_snd']['methods']['test']
        self.assertEqual(fixed['success_rate'], 0.5)
        self.assertEqual(fixed['mean_best_observed_relative_gap_on_success'], 0.2)
        self.assertEqual(summary['linear_flow_special_case']['instance_count'], 1)

    def test_objective_bars_show_relative_gap_and_failure(self):
        import math
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from final_code_test import _draw_group_figure
        cases = [dict(name='toy', results=[
            dict(method='a', status='ok', objective=100.0),
            dict(method='b', status='ok', objective=125.0),
            dict(method='c', status='failed', objective=None)])]
        fig, ax = plt.subplots()
        try:
            _draw_group_figure('fig01_objective_by_instance', 'fixed_charge_snd', cases, ax=ax)
            self.assertEqual(ax.patches[0].get_height(), 0.0)
            self.assertEqual(ax.patches[1].get_height(), 25.0)
            self.assertTrue(math.isnan(ax.patches[2].get_height()))
            self.assertIn('F', [text.get_text() for text in ax.texts])
        finally:
            plt.close(fig)

    def test_publication_figures_write_all_formats(self):
        from final_code_test import generate_benchmark_figures
        with TemporaryDirectory() as directory, patch('final_code_test.FIGURE_DPI', 80):
            report = {
                'schema': 'snd-benchmark-comparison-v1',
                'result_dir': 'synthetic-test',
                'instances': [{
                    'name': 'toy_a', 'nodes': 10, 'arcs': 20, 'commodities': 3,
                    'metadata': {'comparison_group': 'fixed_charge_snd'},
                    'results': [
                        {'method': 'exported_llm_heuristic', 'status': 'ok',
                         'feasible': True, 'objective': 90.0, 'solve_time': 0.1},
                        {'method': 'variable_cost', 'status': 'ok',
                         'feasible': True, 'objective': 100.0, 'solve_time': 0.01},
                    ],
                }, {
                    'name': 'toy_b', 'nodes': 20, 'arcs': 40, 'commodities': 6,
                    'metadata': {'comparison_group': 'fixed_charge_snd'},
                    'results': [
                        {'method': 'exported_llm_heuristic', 'status': 'ok',
                         'feasible': True, 'objective': 180.0, 'solve_time': 0.2},
                        {'method': 'variable_cost', 'status': 'failed',
                         'feasible': False, 'objective': None, 'solve_time': 0.1},
                    ],
                }],
            }
            stamp = '20261019_123456_000001'
            result = generate_benchmark_figures(report, directory, run_stamp=stamp)
            self.assertEqual(Path(result['output_dir']).name, stamp)
            self.assertEqual(set(result['files']), {
                'fig01_objective_by_instance', 'fig02_runtime_by_instance',
                'fig03_relative_gap_heatmap', 'fig04_feasibility_by_scale'})
            for paths in result['files'].values():
                self.assertEqual(set(paths), {'png', 'pdf'})
                for extension in ('png', 'pdf'):
                    self.assertTrue(Path(paths[extension]).exists())
                    self.assertGreater(Path(paths[extension]).stat().st_size, 0)
                    self.assertTrue(Path(paths[extension]).is_relative_to(result['output_dir']))
                    self.assertIn(stamp, Path(paths[extension]).stem)
            self.assertTrue(Path(result['manifest']).exists())
            self.assertTrue(Path(result['index']).exists())
            self.assertTrue((Path(directory) / 'index.html').exists())
            self.assertEqual(len(list(Path(result['output_dir']).rglob('*.png'))), 4)
            self.assertEqual(len(list(Path(result['output_dir']).rglob('*.pdf'))), 4)
            self.assertEqual(list(Path(result['output_dir']).rglob('*.svg')), [])
            first_png = Path(result['files']['fig01_objective_by_instance']['png']).read_bytes()
            other_png = Path(result['files']['fig02_runtime_by_instance']['png']).read_bytes()
            self.assertNotEqual(first_png, other_png)
            # 即使时间戳相同也新建目录，不能覆盖原图或引用其他目录中的文件。
            repeated = generate_benchmark_figures(
                {**report, 'result_dir': 'another-test'}, directory, run_stamp=stamp)
            self.assertFalse(repeated['reused'])
            self.assertNotEqual(result['output_dir'], repeated['output_dir'])
            self.assertEqual(result['data_fingerprint'], repeated['data_fingerprint'])
            manifest = json.loads(Path(repeated['manifest']).read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['sources']), 1)
            self.assertEqual(manifest['generated_assets'], 4)
            self.assertEqual(manifest['last_plot_stamp'], stamp)
            page = Path(repeated['index']).read_text(encoding='utf-8')
            self.assertIn(stamp, page)
            for paths in repeated['files'].values():
                for path in paths.values():
                    self.assertTrue(Path(path).is_relative_to(repeated['output_dir']))
                    self.assertIn(Path(path).relative_to(repeated['output_dir']).as_posix(), page)
            # 耗时变化改变实际报告指纹；质量图的数据指纹不受无关耗时影响。
            changed = deepcopy(report)
            changed['instances'][0]['results'][0]['solve_time'] = 0.9
            updated = generate_benchmark_figures(changed, directory)
            old_manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
            updated_manifest = json.loads(Path(updated['manifest']).read_text(encoding='utf-8'))
            self.assertNotEqual(result['data_fingerprint'], updated['data_fingerprint'])
            for key in ('fig03_relative_gap_heatmap', 'fig01_objective_by_instance',
                        'fig04_feasibility_by_scale'):
                self.assertEqual(updated_manifest['assets'][key]['data_fingerprint'],
                                 old_manifest['assets'][key]['data_fingerprint'])
            # 目标改变必须反映到质量图，而不能沿用旧数据。
            changed['instances'][0]['results'][0]['objective'] = 80.0
            changed_quality = generate_benchmark_figures(changed, directory)
            changed_manifest = json.loads(Path(changed_quality['manifest']).read_text(encoding='utf-8'))
            for key in ('fig03_relative_gap_heatmap', 'fig01_objective_by_instance'):
                self.assertNotEqual(changed_manifest['assets'][key]['data_fingerprint'],
                                    updated_manifest['assets'][key]['data_fingerprint'])
            # 原始快照保持完整。
            self.assertEqual(Path(result['files']['fig01_objective_by_instance']['png']).read_bytes(),
                             first_png)

    def test_figure_stamp_cannot_escape_output_directory(self):
        from final_code_test import generate_benchmark_figures
        report = dict(instances=[dict(name='toy', nodes=2, arcs=1, commodities=1,
                                     metadata=dict(comparison_group='fixed_charge_snd'), results=[])])
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                generate_benchmark_figures(report, directory, run_stamp='../outside')

    def test_report_write_retries_transient_file_lock(self):
        from final_code_test import write_json
        original_replace = Path.replace
        calls = []

        def replace(path, target):
            calls.append(path)
            if len(calls) < 3:
                raise PermissionError('temporary lock')
            return original_replace(path, target)

        with TemporaryDirectory() as directory, patch.object(Path, 'replace', replace), \
                patch('final_code_test.time.sleep'):
            path = Path(directory) / 'report.json'
            write_json(path, dict(value=42))
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), dict(value=42))
            self.assertEqual(len(calls), 3)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_report_write_retains_data_on_persistent_lock(self):
        from final_code_test import write_json
        with TemporaryDirectory() as directory, \
                patch.object(Path, 'replace', side_effect=PermissionError('locked')), \
                patch('final_code_test.time.sleep'):
            path = Path(directory) / 'report.json'
            path.write_text('{"old": true}', encoding='utf-8')
            with self.assertRaises(PermissionError):
                write_json(path, dict(value=42))
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), dict(old=True))
            temporary, = Path(directory).glob('*.tmp')
            self.assertEqual(json.loads(temporary.read_text(encoding='utf-8')), dict(value=42))

    def test_figure_statuses_include_all_failed_skipped_and_pending(self):
        from final_code_test import (_figure_cases, _figure_rows, _feasibility_by_scale,
                                     generate_benchmark_figures)
        cases = [
            dict(name='failed', nodes=150, arcs=10, commodities=20,
                 metadata=dict(comparison_group='linear_flow_special_case'),
                 results=[dict(method='variable_cost', status='failed', objective=None,
                               feasible=False, solve_time=1.0)]),
            dict(name='skipped', nodes=150, arcs=10, commodities=20,
                 metadata=dict(comparison_group='linear_flow_special_case'),
                 input_audit=dict(status='proven_infeasible'), results=[]),
            dict(name='pending', nodes=150, arcs=10, commodities=20,
                 metadata=dict(comparison_group='linear_flow_special_case'), results=[]),
        ]
        report = dict(schema='snd-benchmark-comparison-v1', instances=cases)
        normalized = _figure_cases(report)['linear_flow_special_case']
        self.assertEqual(len(normalized), 3)
        stats = _feasibility_by_scale(normalized, 'variable_cost')[150]
        self.assertEqual(stats, dict(attempted=1, feasible=0, rate=0.0))
        self.assertEqual(_figure_rows(report, group='linear_flow_special_case'), [])
        with TemporaryDirectory() as directory, patch('final_code_test.FIGURE_DPI', 80):
            result = generate_benchmark_figures(report, directory)
            self.assertEqual(len(result['files']), 4)  # 全无可行解时也保留失败/跳过/待执行信息。
            manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertEqual(len(manifest['groups']['linear_flow_special_case']), 3)
            self.assertIn('fig04_feasibility_by_scale', result['files'])

    def test_figures_keep_model_groups_separate(self):
        from final_code_test import generate_benchmark_figures
        cases = [dict(name=group, nodes=10, arcs=20, commodities=1,
                      metadata=dict(comparison_group=group),
                      results=[dict(method='variable_cost', status='ok', feasible=True,
                                    objective=100.0, solve_time=0.1)])
                 for group in ('fixed_charge_snd', 'linear_flow_special_case')]
        with TemporaryDirectory() as directory, patch('final_code_test.FIGURE_DPI', 80):
            result = generate_benchmark_figures(dict(instances=cases), directory)
            self.assertEqual(len(result['files']), 4)  # 模型类别只增加面板，不增加图片数量。
            manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertEqual(set(manifest['groups']), {'fixed_charge_snd', 'linear_flow_special_case'})

    def test_old_report_larger_instances_are_excluded_from_figures(self):
        from final_code_test import generate_benchmark_figures
        report = dict(instances=[dict(name=f'toy{n}', nodes=n, arcs=20, commodities=1,
                      metadata=dict(comparison_group='fixed_charge_snd'), results=[])
                      for n in (10, 200, 300, 500)])
        with TemporaryDirectory() as directory, patch('final_code_test.FIGURE_DPI', 80):
            result = generate_benchmark_figures(report, directory)
            manifest = json.loads(Path(result['manifest']).read_text(encoding='utf-8'))
            self.assertEqual(manifest['excluded_over_node_limit'], ['toy300', 'toy500'])
            self.assertEqual([case['nodes'] for case in manifest['groups']['fixed_charge_snd']], [10, 200])
