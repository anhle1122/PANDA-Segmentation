# Paper decisions (Omar)

## 2026-11-11 — AGGC primary model

**Locked:** Zero-shot PANDA R3 ep6 is the **primary** model in the paper.

- Checkpoint: `outputs/checkpoints/uni2_upernet_raw_opt3_omar6_round3_ep7ref/epoch_006_cancer_0.3488.pth`
- Reason: never trained on SICAP or AGGC; native-raw results show SICAP fine-tuning neither helped nor hurt AGGC much; S1 ZS did not beat the 0.621 SICAP number Omar hoped ZS would beat.
- SICAP-FT (`epoch_006_cancer_0.6893.pth`) → **supplement** only.

Quote (Omar → Anh):
> i expected the zero shot model to beat 0.621 on S1 at native but it didnt. so with everything at native raw it seems fine tuning on SICAP neither helped nor hurt AGGC significantly. lets keep the zero shot model as the primary model in the paper since it didn’t see both datasets and SICAP fine tuning can go in supp or something
