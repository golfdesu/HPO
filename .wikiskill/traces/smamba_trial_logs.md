# Raw Execution Traces: S-Mamba 26-Trial Execution Logs (NVIDIA H100)

**Date**: 2026-09-03 01:27:00 to 06:03:31 (Total Duration: 4h 36m)  
**Hardware**: NVIDIA H100 80GB HBM3  
**Study**: `17_hpo_smamba_pytorch_full`  

### Chronological Trial Log:
```text
[I 2026-09-03 01:27:00,876] A new study created in memory with name: 17_hpo_smamba_pytorch_full
[I 2026-09-03 01:29:48,013] Trial 0 finished with value: 0.004677173519485911 and parameters: {'d_model': 64, 'd_state': 8, 'num_layers': 1, 'dropout_rate': 0.3, 'learning_rate': 0.0010502105436744284, 'weight_decay': 0.0006796578090758161, 'batch_size': 128}. (Duration: 2m 48s)
[I 2026-09-03 01:31:29,590] Trial 1 finished with value: 0.003682394778005976 and parameters: {'d_model': 32, 'd_state': 16, 'num_layers': 1, 'dropout_rate': 0.2, 'learning_rate': 0.00017258215396625024, 'weight_decay': 1.4742753159914662e-05, 'batch_size': 256}. (Duration: 1m 41s)
[I 2026-09-03 01:36:16,760] Trial 2 finished with value: 0.005690965931251889 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 1, 'dropout_rate': 0.3, 'learning_rate': 0.0043709904681305065, 'weight_decay': 0.0017123375973163992, 'batch_size': 256}. (Duration: 4m 47s)
[I 2026-09-03 01:47:12,165] Trial 3 finished with value: 0.003263255930065282 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 2, 'dropout_rate': 0.1, 'learning_rate': 0.0007648565112369955, 'weight_decay': 0.00015375920235481777, 'batch_size': 128}. (Duration: 10m 56s) [GLOBAL BEST]
[I 2026-09-03 01:50:03,721] Trial 4 finished with value: 0.003704866954016242 and parameters: {'d_model': 32, 'd_state': 8, 'num_layers': 1, 'dropout_rate': 0.1, 'learning_rate': 0.00045745782054754043, 'weight_decay': 1.2172958098369984e-05, 'batch_size': 64}. (Duration: 2m 51s)
[I 2026-09-03 01:55:27,805] Trial 5 finished with value: 0.004079283614411556 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 1, 'dropout_rate': 0.05, 'learning_rate': 0.0024290950368254954, 'weight_decay': 0.0006720930050156114, 'batch_size': 128}. (Duration: 5m 24s)
[I 2026-09-03 02:01:10,324] Trial 6 finished with value: 0.003771411383014835 and parameters: {'d_model': 128, 'd_state': 8, 'num_layers': 1, 'dropout_rate': 0.1, 'learning_rate': 0.00173611708903933, 'weight_decay': 0.0003550012525851161, 'batch_size': 64}. (Duration: 5m 43s)
[I 2026-09-03 02:09:50,238] Trial 7 finished with value: 0.003900160445170512 and parameters: {'d_model': 64, 'd_state': 8, 'num_layers': 2, 'dropout_rate': 0.05, 'learning_rate': 0.00015251209898002935, 'weight_decay': 1.3357240411974112e-06, 'batch_size': 64}. (Duration: 8m 40s)
[I 2026-09-03 02:12:14,465] Trial 8 finished with value: 0.005485716129067115 and parameters: {'d_model': 32, 'd_state': 8, 'num_layers': 1, 'dropout_rate': 0.05, 'learning_rate': 0.0037977679442478553, 'weight_decay': 0.0017079750342958238, 'batch_size': 128}. (Duration: 2m 24s)
[I 2026-09-03 02:17:55,128] Trial 9 finished with value: 0.005258057426196179 and parameters: {'d_model': 64, 'd_state': 16, 'num_layers': 1, 'dropout_rate': 0.1, 'learning_rate': 0.0005316714274124606, 'weight_decay': 0.0018709365688887364, 'batch_size': 64}. (Duration: 5m 41s)
[I 2026-09-03 02:46:25,849] Trial 10 finished with value: 0.0035571986844605786 and parameters: {'d_model': 128, 'd_state': 32, 'num_layers': 3, 'dropout_rate': 0.2, 'learning_rate': 0.00036593858966025726, 'weight_decay': 8.77200156873305e-05, 'batch_size': 128}. (Duration: 28m 30s)
[I 2026-09-03 03:11:08,265] Trial 11 finished with value: 0.0036796361576732894 and parameters: {'d_model': 128, 'd_state': 32, 'num_layers': 3, 'dropout_rate': 0.2, 'learning_rate': 0.0004027061919016856, 'weight_decay': 7.069331472481826e-05, 'batch_size': 128}. (Duration: 24m 43s)
[I 2026-09-03 03:31:04,618] Trial 12 finished with value: 0.003387856303067105 and parameters: {'d_model': 128, 'd_state': 32, 'num_layers': 3, 'dropout_rate': 0.2, 'learning_rate': 0.0009199242357311898, 'weight_decay': 9.241634598289258e-05, 'batch_size': 128}. (Duration: 19m 56s)
[I 2026-09-03 03:42:58,837] Trial 13 finished with value: 0.003824235406360139 and parameters: {'d_model': 128, 'd_state': 32, 'num_layers': 2, 'dropout_rate': 0.15, 'learning_rate': 0.0010207554784429087, 'weight_decay': 9.283827996757223e-05, 'batch_size': 128}. (Duration: 11m 54s)
[I 2026-09-03 03:49:14,801] Trial 14 pruned. (Duration: 6m 16s)
[I 2026-09-03 03:56:30,604] Trial 15 finished with value: 0.0038887889548422107 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 3, 'dropout_rate': 0.15, 'learning_rate': 0.0017947623213558606, 'weight_decay': 1.7536894008031622e-05, 'batch_size': 128}. (Duration: 7m 16s)
[I 2026-09-03 04:02:23,149] Trial 16 pruned. (Duration: 5m 53s)
[I 2026-09-03 04:19:08,722] Trial 17 finished with value: 0.0034819986673558195 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 3, 'dropout_rate': 0.25, 'learning_rate': 0.0007161456568704222, 'weight_decay': 4.2640035212433757e-05, 'batch_size': 128}. (Duration: 16m 45s)
[I 2026-09-03 04:21:28,991] Trial 18 pruned. (Duration: 2m 20s)
[I 2026-09-03 04:25:40,731] Trial 19 finished with value: 0.00395550140183217 and parameters: {'d_model': 64, 'd_state': 16, 'num_layers': 3, 'dropout_rate': 0.15, 'learning_rate': 0.0014389096186182106, 'weight_decay': 5.832598988304538e-06, 'batch_size': 256}. (Duration: 4m 12s)
[I 2026-09-03 04:31:56,680] Trial 20 pruned. (Duration: 6m 16s)
[I 2026-09-03 04:48:42,502] Trial 21 finished with value: 0.003632238549315388 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 3, 'dropout_rate': 0.25, 'learning_rate': 0.0006690799484351058, 'weight_decay': 0.000189211001744003, 'batch_size': 128}. (Duration: 16m 46s)
[I 2026-09-03 05:05:28,381] Trial 22 finished with value: 0.0037401386878416786 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 3, 'dropout_rate': 0.25, 'learning_rate': 0.0006600048827278411, 'weight_decay': 3.288365766486133e-05, 'batch_size': 128}. (Duration: 16m 46s)
[I 2026-09-03 05:17:12,503] Trial 23 finished with value: 0.0036887685706061494 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 3, 'dropout_rate': 0.2, 'learning_rate': 0.0012661499987949452, 'weight_decay': 4.145725128821939e-05, 'batch_size': 128}. (Duration: 11m 44s)
[I 2026-09-03 05:33:58,769] Trial 24 finished with value: 0.0036506479051701657 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 3, 'dropout_rate': 0.3, 'learning_rate': 0.0007122445834500785, 'weight_decay': 0.00012479113911099756, 'batch_size': 128}. (Duration: 16m 46s)
[I 2026-09-03 05:44:32,293] Trial 25 finished with value: 0.003739488747999692 and parameters: {'d_model': 128, 'd_state': 16, 'num_layers': 2, 'dropout_rate': 0.25, 'learning_rate': 0.0005457697646271264, 'weight_decay': 5.188799662022923e-05, 'batch_size': 128}. (Duration: 10m 34s)
[I 2026-09-03 06:03:31,794] Trial 26 finished with value: 0.003614513047302581 and parameters: {'d_model': 128, 'd_state': 32, 'num_layers': 3, 'dropout_rate': 0.2, 'learning_rate': 0.0024097492922846215, 'weight_decay': 0.00019038025426783864, 'batch_size': 128}. (Duration: 18m 59s)
```\n