"""Replay named failures and preserve collision context, without changing a policy."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from env.robotaxi import RobotaxiEnv
from ppo import load_checkpoint, forward
from scenarios import VARIANTS, vary_initial_state


def main():
    checkpoint = Path('artifacts/ppo_v2_map2_seed0/checkpoint_1600.npz')
    params, config = load_checkpoint(checkpoint)
    p, initial = RobotaxiEnv.init_params(jax.random.PRNGKey(0), map_id=2)
    rows = json.loads(Path('artifacts/evaluation/map2_final.json').read_text())['per_episode']
    seeds = [r['reset_seed'] for r in rows if r['collision']]
    names = [f'regression_seed_{seed}' for seed in seeds] + ['heading_plus_10deg']
    states = [RobotaxiEnv.reset(jax.random.PRNGKey(seed), p, initial, randomize=True)[1] for seed in seeds]
    states += [vary_initial_state(p, initial, **dict(VARIANTS)['heading_plus_10deg'])]
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states)

    @jax.jit
    def run(states):
        def tick(carry, t):
            states, action = carry
            obs = jax.vmap(lambda s: RobotaxiEnv.get_observation(s, p).as_vector())(states)
            action = jax.lax.cond(t % config['repeat'] == 0,
                lambda: jnp.tanh(forward(params, obs)[0]), lambda: action)
            _, states, _, done, info = jax.vmap(lambda s,a: RobotaxiEnv.step(jax.random.PRNGKey(0),s,a,p))(states, action)
            obstacles = jnp.concatenate([states.static_obstacles, states.kinematic_obstacles], axis=1)
            distances = jnp.linalg.norm(states.agent_pos[:,None,:] - obstacles, axis=-1)
            nearest = jnp.argmin(distances, axis=1)
            closest = jnp.take_along_axis(obstacles,nearest[:,None,None],axis=1)[:,0,:]
            record = (states.agent_pos, states.heading, states.speed, action, info['clearance'],
                      info['collision'], info['goal_reached'], nearest, closest, obs)
            return (states,action), record
        return jax.lax.scan(tick,(states,jnp.zeros((len(names),2))),jnp.arange(p.max_steps_in_episode))[1]

    records = jax.tree.map(np.asarray,run(batch))
    positions, headings, speeds, actions, clearances, collisions, goals, indices, closest, obs = records
    out=Path('artifacts/map2_diagnosis'); out.mkdir(exist_ok=True)
    summaries=[]
    for n,name in enumerate(names):
        t=int(np.argmax(collisions[:,n]))
        summary={'case':name,'collision_step':t+1,'position':positions[t,n].tolist(),
                 'heading':float(headings[t,n]),'speed':float(speeds[t,n]),'action':actions[t,n].tolist(),
                 'clearance':float(clearances[t,n]),'obstacle_index':int(indices[t,n]),
                 'obstacle_type':'static' if indices[t,n]<initial.static_obstacles.shape[0] else 'moving',
                 'obstacle_position':closest[t,n].tolist()}
        summaries.append(summary)
        trace=np.column_stack([np.arange(max(0,t-120),t+1)+1,positions[max(0,t-120):t+1,n],
            headings[max(0,t-120):t+1,n],speeds[max(0,t-120):t+1,n],actions[max(0,t-120):t+1,n],
            clearances[max(0,t-120):t+1,n],obs[max(0,t-120):t+1,n,3:6]])
        np.savetxt(out/(name+'.csv'),trace,delimiter=',',header='step,x,y,heading,speed,throttle,steering,clearance,path_local_x,path_local_y,distance_to_path',comments='')
    (out/'summary.json').write_text(json.dumps({'purpose':'known failures reused as regression; not held-out test',
        'checkpoint':str(checkpoint),'cases':summaries},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
