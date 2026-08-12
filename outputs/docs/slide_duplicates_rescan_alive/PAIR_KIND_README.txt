Rescan pair_kind labels
=======================
NEW            = fresh pair among currently alive slides (not a prior kept not-twin pair)
KEPT_NOT_TWIN  = both sides were previously marked not-twins and kept (you already reviewed)
MIXED          = exactly one side was a prior kept not-twin

Files:
  galleries/safe_pairs/pair_index_annotated.csv
  galleries/safe_pairs/pair_index_NEW_only.csv
  galleries/safe_pairs/pair_index_KEPT_NOT_TWIN_only.csv
  galleries/lower_iou/pair_index_*.csv
  galleries/multi_clusters/cluster_index_annotated.csv

Gallery PNGs: re-render job adds green/orange/blue badges on each row.
