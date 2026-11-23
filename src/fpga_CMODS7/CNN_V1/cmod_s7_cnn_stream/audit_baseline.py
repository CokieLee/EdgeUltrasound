#!/usr/bin/env python3
"""Compare the known-good original project with the supplied baseline; no rebuild."""
import argparse
from collections import Counter, deque
import difflib
import hashlib
import json
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def compare_files(original, baseline):
    result = {'original': str(original), 'baseline': str(baseline),
              'original_exists': original.is_file(), 'baseline_exists': baseline.is_file()}
    if not (original.is_file() and baseline.is_file()):
        return result
    left, right = original.read_bytes(), baseline.read_bytes()
    result.update({'identical_bytes': left == right,
                   'identical_after_crlf_normalization': left.replace(b'\r\n', b'\n') == right.replace(b'\r\n', b'\n'),
                   'original_sha256': digest(left), 'baseline_sha256': digest(right)})
    if left != right:
        changes = list(difflib.unified_diff(left.decode('utf-8', errors='replace').splitlines(),
                    right.decode('utf-8', errors='replace').splitlines(),
                    fromfile=str(original), tofile=str(baseline), n=3))
        result['diff_first_100_lines'] = changes[:100]
        result['diff_truncated'] = len(changes) > 100
    return result


def summarize_netlist(path):
    result = {'path': str(path), 'exists': path.is_file()}
    if not path.is_file():
        return result
    raw = path.read_bytes()
    result['sha256'] = digest(raw)
    try:
        data = json.loads(raw)
        types, parameters, names = Counter(), Counter(), []
        tops, modules, source_examples = [], {}, set()
        for module_name, module in data['modules'].items():
            attributes = module.get('attributes', {})
            top = attributes.get('top', '0')
            if str(top).strip('0') == '1':
                tops.append(module_name)
            cells = module.get('cells', {})
            if not cells:
                continue
            modules[module_name] = len(cells)
            for cell_name, cell in cells.items():
                cell_type = cell['type']
                types[cell_type] += 1
                signature = json.dumps([cell_type, cell.get('parameters', {})], sort_keys=True)
                parameters[signature] += 1
                names.append(module_name + '/' + cell_name)
                for part in cell.get('attributes', {}).get('src', '').split('|'):
                    if 'rtl/' in part.replace('\\', '/'):
                        source_examples.add(part)
        result.update({'creator': data.get('creator'), 'top_modules': sorted(tops),
            'modules_with_cells': modules, 'total_cells': sum(types.values()),
            'cell_types': dict(sorted(types.items())),
            'cell_type_parameter_multiset_sha256': digest(json.dumps(sorted(parameters.items())).encode()),
            'cell_names_sha256': digest(json.dumps(sorted(names)).encode()),
            'source_attribute_examples': sorted(source_examples)[:25]})
    except (ValueError, KeyError, TypeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    return result


def summarize_log(path):
    result = {'path': str(path), 'exists': path.is_file()}
    if not path.is_file():
        return result
    first, last = [], deque(maxlen=12)
    versions, markers = [], []
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            text = line.rstrip('\n')
            if len(first) < 12:
                first.append(text)
            last.append(text)
            if ('Yosys ' in text or 'Next Generation Place' in text) and len(versions) < 4:
                versions.append(text)
            if any(token in text for token in ('Command line:', 'Program finished normally',
                                               'max placement attempts per cell', 'ERROR:')):
                if len(markers) < 20:
                    markers.append(text)
    result.update({'first_lines': first, 'last_lines': list(last),
                   'version_lines': versions, 'markers': markers})
    return result


def audit(original, stream):
    pairs = {name: (original / 'rtl' / name,
                   stream / ('baseline/rtl' if name in ('cnn_engine.sv', 'top.sv') else 'rtl') / name)
             for name in ('cnn_engine.sv', 'top.sv', 'uart_rx.sv', 'uart_tx.sv')}
    pairs['pin_constraints'] = (original / 'cmod_s7_cnn.xdc', stream / 'cmod_s7_cnn_stream.xdc')
    pairs['build_script'] = (original / 'build.sh', stream / 'build.sh')
    files = {name: compare_files(*paths) for name, paths in pairs.items()}
    good = summarize_netlist(original / 'build/top.json')
    baseline = summarize_netlist(stream / 'build/top_baseline.json')
    relation = {}
    if 'cell_types' in good and 'cell_types' in baseline:
        relation = {'identical_json_bytes': good['sha256'] == baseline['sha256'],
            'same_cell_type_counts': good['cell_types'] == baseline['cell_types'],
            'same_cell_type_parameter_multiset': good['cell_type_parameter_multiset_sha256'] == baseline['cell_type_parameter_multiset_sha256'],
            'same_cell_names': good['cell_names_sha256'] == baseline['cell_names_sha256'],
            'cell_type_count_changes': {key: {'original': good['cell_types'].get(key, 0),
                                             'baseline': baseline['cell_types'].get(key, 0)}
                for key in sorted(set(good['cell_types']) | set(baseline['cell_types']))
                if good['cell_types'].get(key, 0) != baseline['cell_types'].get(key, 0)}}
    return {'original_project': str(original), 'stream_project': str(stream),
        'files': files, 'original_netlist': good, 'baseline_netlist': baseline,
        'netlist_comparison': relation,
        'logs': {name: summarize_log(path) for name, path in {
            'original_synthesis': original / 'build/synth.log',
            'baseline_synthesis': stream / 'build/baseline_synth.log',
            'original_pnr': original / 'build/pnr.log',
            'baseline_pnr': stream / 'build/baseline_pnr.log'}.items()},
        'current_shell_overrides_not_verified_build_environment': {
            key: os.environ.get(key) for key in ('NEXTPNR', 'CHIPDB', 'FPGA_AS', 'XRAY_DB')},
        'limits': [
            'Cell counts/parameter multisets are diagnostics, not a connectivity or equivalence proof.',
            'Different JSON hashes can reflect names, source paths, bit IDs or metadata rather than behavior.',
            'The original build directory must retain the actual successful synthesis JSON/logs.',
            'Current shell overrides do not reveal changes made by export.sh inside a prior build.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', required=True, type=Path,
                        help='Actual original project that successfully routed')
    parser.add_argument('--stream', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('baseline_comparison.json'))
    args = parser.parse_args()
    original, stream = args.original.resolve(), args.stream.resolve()
    if not original.is_dir() or not stream.is_dir():
        parser.error('Both project directories must exist; adjust --original/--stream paths')
    result = audit(original, stream)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for name, comparison in result['files'].items():
        if not comparison['original_exists'] or not comparison['baseline_exists']:
            status = 'MISSING'
        else:
            status = 'IDENTICAL' if comparison['identical_bytes'] else 'DIFFERENT'
        print(f'{name}: {status}')
    print('Original netlist tops:', result['original_netlist'].get('top_modules', 'unavailable'))
    print('Baseline netlist tops:', result['baseline_netlist'].get('top_modules', 'unavailable'))
    print('Netlist comparison:', json.dumps(result['netlist_comparison']))
    print('Saved:', args.output.resolve())
    print('No synthesis, placement, programming, or project source modification was performed.')


if __name__ == '__main__':
    main()
