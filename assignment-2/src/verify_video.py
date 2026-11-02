"""Decode the beginning, middle and end of an actual rollout video."""
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('video',type=Path)
    args=parser.parse_args()
    reader=imageio.get_reader(args.video)
    count=reader.count_frames(); meta=reader.get_meta_data()
    for index in [0,count//2,count-1]:
        frame=reader.get_data(index)
        assert frame.shape==(800,1280,3),frame.shape
    reader.close()
    episodes=json.loads(args.video.with_suffix('.json').read_text())['episodes']
    assert all(r['goal_reached'] and not r['collision'] and not r['truncated']
               and r['final_speed']<=.001 and r['final_distance']<=1 for r in episodes)
    result={'frames':count,'fps':meta['fps'],'duration_seconds':count/meta['fps'],
            'decoded_indices':[0,count//2,count-1],'frame_shape':[800,1280,3],
            'parked_episodes':len(episodes)}
    args.video.with_name(args.video.stem+'-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
