find dataset -type f -iname '*.heic' -print0 | while IFS= read -r -d $'\0' f; do
  out="${f%.*}.jpg"
  sips -s format jpeg "$f" --out "$out"
done