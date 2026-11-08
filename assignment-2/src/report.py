"""Verify the human-written report against saved measurements; never overwrite it."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote


def table_rows(text):
    return [[cell.strip() for cell in line.strip().strip('|').split('|')]
            for line in text.splitlines() if line.lstrip().startswith('|')]


def verify(report):
    root=report.parent
    text=report.read_text(encoding="utf-8")
    rows=table_rows(text)
    data=json.loads((root/'artifacts/evaluation/summary.json').read_text(encoding="utf-8"))['results']
    results={r['name']:r for r in data}
    checked=[]

    def require_row(cells,label):
        if cells not in rows:
            raise ValueError(f'Report row differs from saved evidence ({label}): {cells}')
        checked.append(label)

    labels={0:'0 — объезд стены',1:'1 — дополнительное препятствие',
            2:'2 — сложный поворот с трафиком',3:'3 — открытая площадка'}
    for map_id,label in labels.items():
        r=results[f'map{map_id}_final']
        require_row([label,f"{r['successes']}/{r['episodes']}",str(r['collisions']),
                     str(r['timeouts']),f"{r['episode_sim_seconds_mean']:.2f} с"],f'final map {map_id}')
    current = json.loads((root/'artifacts/recheck/summary.json').read_text(encoding="utf-8"))
    for map_id in labels:
        old32, old64 = [
            next(row for row in current['regression'][mode] if row['name'] == f'map{map_id}_final')
            for mode in ('float32', 'float64')
        ]
        fresh = [row for row in current['holdout'] if row['map'] == map_id]
        n, k, c, t = (sum(row[key] for row in fresh)
                      for key in ('episodes', 'successes', 'collisions', 'timeouts'))
        require_row([str(map_id), f"{old32['successes']}/{old32['episodes']}",
                     f"{old64['successes']}/{old64['episodes']}", f"{k}/{n}", str(c), str(t)],
                    f'recheck map {map_id}')

    for name,label in [('map0_base','За приближение к цели по прямой'),
                       ('map0_shaped500','За продвижение по маршруту и подходящую скорость')]:
        r=results[name]
        require_row([label,f"{r['successes']}/{r['episodes']}",str(r['collisions']),str(r['timeouts'])],name)

    for update in [600,800]:
        r=json.loads((root/f'artifacts/map2_safe_b_dev{update}.json').read_text(encoding="utf-8"))['results'][0]
        require_row([f'После {update} обновлений дообучения',f"{r['successes']}/{r['episodes']}",
                     str(r['collisions']),str(r['timeouts'])],f'checkpoint selection {update}')
    r6=json.loads((root/'artifacts/map2_safe_b_dev600.json').read_text(encoding="utf-8"))['results'][0]
    frequency={6:r6}
    frequency.update({r['evaluation_repeat']:r for r in json.loads((root/'artifacts/map2_repeat_ablation_dev.json').read_text(encoding="utf-8"))['results']})
    for repeat,hz,seconds in [(6,5,'0.20 секунды'),(3,10,'0.10 секунды'),(1,30,'0.033 секунды')]:
        r=frequency[repeat]
        require_row([f'{hz} решений в секунду',seconds,f"{r['successes']}/{r['episodes']}",
                     str(r['collisions']),str(r['timeouts'])],f'control {hz}Hz')

    ci=json.loads((root/'artifacts/ci_results.json').read_text(encoding="utf-8"))
    assert ci['passed']
    for job in ci['jobs']:
        label='Ubuntu 24.04' if job['os']=='ubuntu-24.04' else 'Windows Server 2022'
        require_row([label,f"{job['tests']}/{job['tests']}",f"{job['map2_parked']}/{job['map2_cases']} парковок",
                     f"{job['endpoint_parked']}/{job['endpoint_cases']} парковок",'Записано и декодировано'],label)

    for map_id in range(4):
        r=results[f'map{map_id}_stochastic']
        require_row([str(map_id),f"{r['successes']}/{r['episodes']}",str(r['collisions']),str(r['timeouts'])],f'stochastic map {map_id}')
    assert text.count('<details>')==text.count('</details>')==text.count('<summary>')==text.count('</summary>')
    assert '\\n' not in text,'Literal newline escape in prose'
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if '://' not in link and not link.startswith('#'):
            target=root/unquote(link.split('#')[0])
            assert target.exists(),target
    assert 'Автоматической остановки по качеству в реализации нет' in text
    assert results['map2_final']['episodes']==512 and results['map2_final']['seed_start']==50000
    assert len(data)==35 and sum(r['episodes'] for r in data)==4644
    return {'status':'passed','checked_rows':len(checked),'checks':checked,
            'report':'REPORT.md','text_modified':False,
            'scope':'saved measurements, local links and disclosure of manual stopping budgets'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,default=Path('REPORT.md'))
    args=parser.parse_args()
    print(json.dumps(verify(args.report),ensure_ascii=False,indent=2))
