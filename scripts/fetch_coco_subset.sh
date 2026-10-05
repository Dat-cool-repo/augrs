#!/usr/bin/env bash
# Download COCO val2017 annotations (~240 MB zip) and the first 300 val images (~50 MB)
# into $AUGRS_DATA (default: <repo>/data, git-ignored). Idempotent. COCO images are licensed
# by their owners (see https://cocodataset.org/#termsofuse) and are never committed to this repo.
set -euo pipefail
source "$(dirname "$0")/env.sh"
D=$AUGRS_DATA
mkdir -p "$D/val2017"
cd "$D"
if [ ! -f annotations/instances_val2017.json ]; then
  [ -f annotations_trainval2017.zip ] || curl -sS -L -o annotations_trainval2017.zip \
      http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  python3 -c "import zipfile; zipfile.ZipFile('annotations_trainval2017.zip').extract('annotations/instances_val2017.json')"
fi
python3 - "$D" <<'EOF'
import json, sys
d = sys.argv[1]
coco = json.load(open(f'{d}/annotations/instances_val2017.json'))
imgs = sorted(coco['images'], key=lambda x: x['id'])[:300]
ids = {i['id'] for i in imgs}
anns = [a for a in coco['annotations'] if a['image_id'] in ids and not a.get('iscrowd', 0)]
json.dump({'images': imgs, 'annotations': anns, 'categories': coco['categories']}, open(f'{d}/subset300.json', 'w'))
open(f'{d}/files.txt', 'w').write('\n'.join(i['file_name'] for i in imgs))
print(len(imgs), 'images,', len(anns), 'annotations')
EOF
cd val2017
xargs -P 8 -I{} sh -c '[ -f {} ] || curl -sS -o {} http://images.cocodataset.org/val2017/{}' < ../files.txt
echo "$(ls | wc -l) images in $D/val2017"
