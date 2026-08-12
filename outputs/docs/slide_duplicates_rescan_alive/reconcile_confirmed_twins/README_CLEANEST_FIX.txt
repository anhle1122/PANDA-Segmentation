CLEAREST FIX (recommended)
==========================

Truth order (highest wins):
1) You said NOT TWINS → always KEEP both
2) You / policy said TWINS (safe unmarked, lower unmarked, explicit multi drop, today's callouts) → DROP
3) Auto multi-greedy / bulk Friday restore noise → DO NOT trust alone

Right now:
- Alive slides: 3855
- HARD confirmed-drop IDs that should be gone: 831
- Already gone: 826
- HARD LEAKS still alive (safe to re-drop): 0
- SOFT alive from greedy/gallery only (mostly Friday "cluster safe" restores): 24 — review only if you want

Do this:
1. Apply HARD_LEAKS_re_drop_these.csv (one click)
2. Freeze a canonical drop list: confirmed_twin_drop_ids_canonical.txt
3. Re-run twin rescan on cleaned alive set (new folder) so galleries aren't mixed with resurrected twins

Files:
- HARD_LEAKS_re_drop_these.csv
- SOFT_alive_do_not_auto_drop.csv
- HARD_all_confirmed_drop_status.csv
