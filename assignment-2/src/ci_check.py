"""Native CI quality gate for the actual frozen model artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import numpy as np

from evaluate_rl import evaluate
from ppo import load_checkpoint
from make_rl_video import checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes',type=int,default=128)
    parser.add_argument('--seed',type=int,default=60000)
    parser.add_argument('--output',type=Path,default=Path('artifacts/ci-quality.json'))
    args=parser.parse_args()
    rows=[]
    training_smoke=None
    smoke=Path('artifacts/ci-training/checkpoint.npz')
    if smoke.exists():
        trained,metadata=load_checkpoint(smoke)
        initial,_=load_checkpoint(smoke.with_name('untrained.npz'))
        assert metadata['completed_updates']==1
        assert any(not np.array_equal(a,b) for a,b in zip(jax.tree.leaves(trained['actor']),jax.tree.leaves(initial['actor'])))
        training_smoke={'completed_updates':1,'actor_changed':True}
    for map_id in range(4):
        path=checkpoint_path(map_id)
        params,config=load_checkpoint(path)
        result,_=evaluate(params,config,map_id,episodes=1,seed=args.seed,heading_noise=0.,randomize=False)
        result.update(name=f'map{map_id}_canonical',checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(result)
    path=checkpoint_path(2)
    params,config=load_checkpoint(path)
    for name,variants in [('map2_fresh_randomized',False),('map2_endpoint_variants',True)]:
        result,_=evaluate(params,config,2,episodes=args.episodes,seed=args.seed,randomize=True,variants=variants)
        result.update(name=name,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(result)
    document={'purpose':'native frozen parking quality gate plus separate one-update training smoke; frozen models never retrained',
              'results':rows,'training_smoke':training_smoke,
              'passed':all(r['successes']==r['episodes'] and r['collisions']==r['timeouts']==0 for r in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(document,indent=2)+'\n')
    for r in rows:
        print(f"{r['name']}: {r['successes']}/{r['episodes']} parked; {r['collisions']} collisions; {r['timeouts']} timeouts",flush=True)
    if not document['passed']:
        raise SystemExit('Frozen model quality gate failed; see per-episode evidence')


if __name__=='__main__':
    main()
