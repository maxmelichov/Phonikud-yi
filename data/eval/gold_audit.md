# Gold audit with the lattice ear (`runs/final/full2/best_acc`)

For every word of every gold clip the ear scored the gold reading against every alternative it could prefer, in two tiers. **Tier A (verified/menu)**: the gold segment with another stress, the word's other dictionary variants, its readings-menu entries — readings the ear was trained to tell apart; a dispute here at margin >= 2 nats (`menu_margin <= -2`) is a gold label to re-check. **Tier B (graph-only)**: open-slot alternatives (a/ɔ/u, aj/ej, ɔj/oʊ, i/u, ɛ/ə, final devoicing) the corpus never attested for the word; the ear is not calibrated on those, so only margins >= 5 nats are listed, as *possible missing readings*. `gold_train` was in the ear's training data (a dispute there survived being trained on); the val splits are held out.


## gold_val_eps: 11,390 words, 3,594 clips

- the ear's free choice (all tiers) is the gold segment on 81.7% of words, the gold reading incl. stress on 81.5%
- **Tier A disputes (verified/menu alternative, margin >= 2): 69 (0.61%)** — 3,634 words had a tier-A alternative at all
- stress disputes (same segment, margin >= 2): 20 of 611 polysyllables (3.27%)
- Tier B (graph-only alternative, margin >= 5): 67 (0.59%)

### Tier A: word types whose verified/menu reading the ear disputes (>= 3 clips, >= 10% of the type's clips, margin >= 2) — 3 types

| word (key) | disputed / clips | gold segment | ear wants |
|---|---:|---|---|
| דארפנ | 5 / 8 | daːfn ×5 | darfn ×5 |
| יומ | 3 / 15 | jɔm ×3 | jɔjm ×3 |
| געהערט | 3 / 19 | ɡəhɛrt ×3 | ɡəhirt ×3 |

### Tier B: word types where the ear wants a reading outside the verified set (>= 3 clips, >= 10% of the type's clips, margin >= 5) — 1 types

| word (key) | disputed / clips | gold segment | ear wants |
|---|---:|---|---|
| פריינד | 3 / 4 | fraːnd ×3 | frajnt ×3 |

### Stress the ear disputes (word, gold reading) with >= 3 clips

| word | gold | clips disputed / clips with this gold | ear's stress |
|---|---|---:|---|
| בחורימ | buxˈirim | 5 / 5 | bˈuxirim ×4, bˈuxurim ×1 |

### Strongest tier-A clip-level disputes

| clip | word | gold | ear | margin |
|---|---|---|---|---:|
| 129214-00048-000219-0003477.npy | מ'קען | mˈɛkən | mkɛn | -20.4 |
| 153722-00026-000158-0002499.npy | האט | hɔt | hut | -9.1 |
| 39584-00038-000174-0002766.npy | כמעט | kəmˈat | kˈɛmat | -6.3 |
| 129214-00006-000221-0003527.npy | די | də | di | -5.1 |
| 63384-00022-000218-0003470.npy | דאַרפֿן | daːfn | dafn | -4.8 |
| 98150-00037-000050-0000809.npy | אויך | oʊx | ɔjx | -4.8 |
| 41621-00051-000232-0003719.npy | זאגט | zuɡt | zukt | -4.7 |
| 82299-00027-000209-0003332.npy | פרשת | pˈarʃis | pˈarʃəs | -4.2 |
| 98150-00018-000177-0002812.npy | אגב | ˈaɡəv | aɡˈav | -4.1 |
| 41621-00062-000117-0001813.npy | דאלער | dˈɔlər | dɔlˈar | -4.1 |
| 39584-00065-000226-0003616.npy | האט | hɔt | hut | -4.0 |
| 161701-00023-000018-0000281.npy | דאַרפֿן | daːfn | dafn | -4.0 |
| 41621-00046-000205-0003259.npy | דארפן | daːfn | dafn | -3.8 |
| 98150-00009-000143-0002250.npy | האב | hɔb | hub | -3.7 |
| 129214-00019-000116-0001792.npy | די | də | di | -3.7 |
| 140507-00063-000131-0002059.npy | טוט | tit | tut | -3.6 |
| 140507-00030-000055-0000867.npy | אן | un | an | -3.5 |
| 69088-00016-000085-0001326.npy | די | də | di | -3.5 |
| 98150-00002-000006-0000091.npy | האט | hɔt | hut | -3.5 |
| 153722-00032-000020-0000315.npy | די | də | di | -3.5 |
| 41621-00003-000238-0003824.npy | די | də | di | -3.5 |
| 76143-00053-000097-0001511.npy | לאָמיר | lˈumir | lˈɔmir | -3.5 |
| 153722-00039-000029-0000452.npy | האט | hɔt | hut | -3.5 |
| 153722-00012-000079-0001242.npy | זאָגט | zuɡt | zukt | -3.5 |
| 82299-00038-000066-0001029.npy | די | di | də | -3.4 |
| 153722-00001-000091-0001426.npy | ברוך | bˈɔrux | bˈurəx | -3.4 |
| 161701-00023-000018-0000283.npy | דאַרפֿן | daːfn | dafn | -3.3 |
| 48490-00103-000225-0003591.npy | אייך | aːx | ax | -3.3 |
| 48490-00090-000060-0000956.npy | האט | hɔt | hut | -3.3 |
| 63384-00034-000012-0000184.npy | דאַרפֿן | daːfn | dafn | -3.2 |

## gold_val_words: 16,548 words, 3,598 clips

- the ear's free choice (all tiers) is the gold segment on 78.5% of words, the gold reading incl. stress on 78.0%
- **Tier A disputes (verified/menu alternative, margin >= 2): 135 (0.82%)** — 5,877 words had a tier-A alternative at all
- stress disputes (same segment, margin >= 2): 100 of 1,379 polysyllables (7.25%)
- Tier B (graph-only alternative, margin >= 5): 141 (0.85%)

### Tier A: word types whose verified/menu reading the ear disputes (>= 3 clips, >= 10% of the type's clips, margin >= 2) — 2 types

| word (key) | disputed / clips | gold segment | ear wants |
|---|---:|---|---|
| דארפנ | 27 / 42 | daːfn ×27 | darfn ×27 |
| כמעט | 14 / 71 | kəmat ×14 | kimat ×14 |

### Tier B: word types where the ear wants a reading outside the verified set (>= 3 clips, >= 10% of the type's clips, margin >= 5) — 4 types

| word (key) | disputed / clips | gold segment | ear wants |
|---|---:|---|---|
| פרעזידענט | 17 / 40 | prəzidɛnt ×17 | prɛzidɛnt ×12, prɛzidənt ×5 |
| פריינד | 8 / 15 | fraːnd ×8 | frajnt ×4, frund ×1, frant ×1 |
| ארויפ | 10 / 66 | aroʊf ×10 | arɔjp ×8, arɔjf ×2 |
| רוב | 7 / 37 | rɔv ×7 | ruv ×6, raf ×1 |

### Stress the ear disputes (word, gold reading) with >= 3 clips

| word | gold | clips disputed / clips with this gold | ear's stress |
|---|---|---:|---|
| בחורימ | buxˈirim | 35 / 41 | bˈuxirim ×27, bˈuxurim ×6 |
| אביסל | abˈisl | 23 / 46 | ˈabisl ×14, ˈɔbisl ×7 |
| אנגעהויבנ | ˈunɡəhɔjbn | 16 / 43 | inɡəhˈɔjbn ×5, unɡəhˈɔjbn ×5 |
| כמעט | kimˈat | 3 / 25 | kˈimat ×1, kˈumɔt ×1 |
| אגב | ˈaɡav | 3 / 4 | aɡˈaf ×2, aɡˈav ×1 |

### Strongest tier-A clip-level disputes

| clip | word | gold | ear | margin |
|---|---|---|---|---:|
| 56288-00023-002483-0023072.npy | דארפן | daːfn | darfn | -9.9 |
| 39929-00083-001788-0013942.npy | געוואלט | ɡəvˈald | ɡəvˈɔlt | -9.0 |
| 56288-00023-002483-0023073.npy | דארפן | daːfn | dɔfn | -6.7 |
| 95587-00019-002622-0024711.npy | דארפן | daːfn | dɔfn | -6.6 |
| 99961-00050-002357-0021472.npy | רבי | rˈɛbə | rˈɛbi | -6.1 |
| 66679-00099-002423-0022308.npy | אן | un | an | -5.9 |
| 52378-00045-001283-0006295.npy | דארפֿן | daːfn | dafn | -5.8 |
| 46996-00014-002103-0018235.npy | דארפן | daːfn | dafn | -5.6 |
| 118354-00041-001281-0006252.npy | דארפן | daːfn | dafn | -5.6 |
| 118770-00014-001306-0006666.npy | דאַרפֿן | daːfn | dafn | -5.5 |
| 118770-00014-001306-0006667.npy | דאַרפֿן | daːfn | dafn | -5.3 |
| 106917-00028-002610-0024577.npy | דאַרפֿן | daːfn | dafn | -5.3 |
| 99961-00046-001826-0014452.npy | דארפֿן | daːfn | dafn | -5.2 |
| 147785-00014-001193-0004899.npy | קיין | kan | kajn | -5.1 |
| 92835-00006-001511-0009748.npy | דאַרפֿן | daːfn | dafn | -5.0 |
| 157051-00043-001962-0016375.npy | קיין | kajn | kan | -5.0 |
| 116273-00058-002087-0017988.npy | דארפן | daːfn | dafn | -4.9 |
| 158361-00040-002787-0026699.npy | יום | jɔjm | jɔm | -4.8 |
| 48872-00008-002937-0028447.npy | דאַרפֿן | daːfn | dafn | -4.8 |
| 102536-00070-002273-0020463.npy | דארפן | daːfn | dafn | -4.5 |
| 99961-00040-002358-0021495.npy | די | də | di | -4.4 |
| 67630-00003-002363-0021549.npy | דארפן | daːfn | dafn | -4.3 |
| 78837-00091-001353-0007430.npy | כמעט | kəmˈat | kimˈat | -4.2 |
| 48013-00011-002667-0025296.npy | דארפן | daːfn | dafn | -4.2 |
| 86944-00012-002057-0017560.npy | טאג | tuk | tuɡ | -4.1 |
| 90570-00073-002751-0026293.npy | כ'האב | xɔb | xhɔb | -4.1 |
| 39286-00137-002371-0021653.npy | דארפן | daːfn | dafn | -4.1 |
| 99506-00068-002206-0019583.npy | האט | hɔt | hut | -4.0 |
| 72395-00072-001654-0011899.npy | כמעט | kəmˈat | kɛmˈat | -3.9 |
| 71234-00032-002875-0027729.npy | דארפן | daːfn | dafn | -3.9 |

## gold_train: 274,448 words, 77,592 clips

- the ear's free choice (all tiers) is the gold segment on 85.2% of words, the gold reading incl. stress on 85.1%
- **Tier A disputes (verified/menu alternative, margin >= 2): 1,002 (0.37%)** — 92,246 words had a tier-A alternative at all
- stress disputes (same segment, margin >= 2): 178 of 17,149 polysyllables (1.04%)
- Tier B (graph-only alternative, margin >= 5): 1,019 (0.37%)

### Tier A: word types whose verified/menu reading the ear disputes (>= 3 clips, >= 10% of the type's clips, margin >= 2) — 0 types

| word (key) | disputed / clips | gold segment | ear wants |
|---|---:|---|---|

### Tier B: word types where the ear wants a reading outside the verified set (>= 3 clips, >= 10% of the type's clips, margin >= 5) — 3 types

| word (key) | disputed / clips | gold segment | ear wants |
|---|---:|---|---|
| פינ | 8 / 16 | pin ×8 | fin ×8 |
| אפ | 44 / 417 | ɔp ×44 | af ×30, uf ×8, ap ×3 |
| אפאר | 4 / 37 | apur ×3, apɔr ×1 | afar ×2, afir ×2 |

### Stress the ear disputes (word, gold reading) with >= 3 clips

| word | gold | clips disputed / clips with this gold | ear's stress |
|---|---|---:|---|
| אסאכ | ˈasax | 21 / 72 | asˈax ×12, asˈux ×4 |
| מסביר | mazbˈir | 16 / 284 | mˈazbir ×8, mˈuzbir ×4 |
| וויליאמסבורג | vˈiljamsburɡ | 14 / 24 | viljamsbˈurɡ ×10, vuljˈamsburk ×1 |
| ספרימ | sfˈurim | 11 / 343 | sfirˈim ×4, sfarˈim ×3 |
| השמ | haʃˈɛm | 10 / 525 | hˈɔʃɛm ×4, hˈaʃəm ×1 |
| פארוואס | farvˈus | 9 / 289 | fˈarvis ×2, fˈurvus ×2 |
| גארנישט | ɡˈurniʃt | 7 / 589 | ɡurnˈiʃt ×3, ɡurnˈuʃt ×2 |
| אזא | azˈa | 7 / 1060 | ˈaza ×5, uzˈa ×1 |
| אמאל | amˈul | 6 / 592 | ˈamil ×3, ˈɔmil ×2 |
| ארומ | arˈim | 5 / 195 | ˈarim ×4, urˈum ×1 |
| אראפ | arˈup | 5 / 462 | ˈuruf ×1, ˈɔrip ×1 |
| אינמיטנ | ˈinmitn | 5 / 79 | inmˈitn ×3, inmˈutn ×1 |
| אפילו | afˈilə | 4 / 543 | ˈafilə ×3, ˈufilɛ ×1 |
| ניגונימ | niɡˈinim | 4 / 397 | nˈiɡinim ×2, niɡˈunim ×1 |
| האסטו | hˈusti | 4 / 220 | hastˈi ×1, histˈu ×1 |
| אזוי | azˈɔj | 4 / 3154 | ˈazɔj ×2, uzˈɔj ×1 |
| פאמיליע | fˈamiliə | 3 / 10 | famˈiliə ×1, familˈiə ×1 |
| אוועק | avˈɛk | 3 / 496 | ˈavɛk ×2, ˈuvɛk ×1 |
| אברהמ | ˈavrum | 3 / 495 | avrˈim ×2, avrˈum ×1 |
| ארויס | arˈɔjs | 3 / 422 | ˈarɔjs ×3 |

### Strongest tier-A clip-level disputes

| clip | word | gold | ear | margin |
|---|---|---|---|---:|
| 104192-00020-008207-0065572.npy | מ'קען | mˈɛkən | mɛkˈɛn | -8.9 |
| 113370-00032-005778-0051243.npy | אים | ejm | im | -8.0 |
| 69999-00051-006682-0057179.npy | אַנדערש | ˈandiʃ | ˈandərʃ | -7.7 |
| 63743-00041-002465-0022828.npy | אפשר | ˈɛfʃər | ɛfʃˈir | -7.7 |
| 109757-00059-016235-0091462.npy | וויפיל | vˈifil | vifl | -7.5 |
| 103740-00008-001960-0016340.npy | ברוך | bˈɔrux | bˈurəx | -7.4 |
| 45082-00004-001897-0015486.npy | אראפ | arˈɔp | arˈup | -7.3 |
| 142113-00049-002378-0021732.npy | יעקב | jˈankəf | jˈankəv | -6.8 |
| 150093-00083-005816-0051518.npy | אויפן | ɔfn | afn | -6.8 |
| 50193-00083-006152-0053744.npy | אויפן | ɔfn | afn | -6.7 |
| 52378-00045-001283-0006299.npy | רבי | rˈɛbi | rˈɛbə | -6.4 |
| 47736-00001-002675-0025398.npy | אפשר | ˈɛfʃir | ˈɛfʃər | -6.2 |
| 71234-00037-002837-0027279.npy | די | də | di | -6.2 |
| 40335-00074-002884-0027842.npy | געזאגט | ɡəzˈuɡt | ɡəzˈukt | -6.1 |
| 51117-00021-001912-0015686.npy | איד | id | jid | -6.1 |
| 122530-00015-004243-0040365.npy | דו | di | du | -6.1 |
| 51484-00113-003823-0037016.npy | אים | ejm | im | -6.0 |
| 97696-00001-013990-0086761.npy | וויפיל | vˈifil | vifl | -6.0 |
| 51117-00062-005561-0049702.npy | געזאגט | ɡəzˈuɡt | ɡəzˈukt | -5.9 |
| 132235-00010-001527-0009944.npy | לערנען | lˈirnən | lˈɛrnən | -5.9 |
| 102168-00039-004490-0042316.npy | געזאגט | ɡəzˈuɡt | ɡəzˈukt | -5.8 |
| 91348-00049-004701-0043948.npy | ווייטער | vˈaːtər | vatˈɛr | -5.8 |
| 50595-00094-013549-0085614.npy | וויפיל | vˈifil | vifl | -5.8 |
| 66679-00075-007792-0063558.npy | ביז | biz | bis | -5.7 |
| 84352-00040-005416-0048700.npy | אויך | oʊx | ɔjx | -5.6 |
| 127667-00061-003329-0032632.npy | אויפֿן | ɔfn | afn | -5.6 |
| 66679-00027-008325-0066227.npy | זאגט | zukt | zuɡt | -5.5 |
| 102168-00010-003964-0038184.npy | אויך | oʊx | ɔjx | -5.5 |
| 120082-00025-007657-0062817.npy | די | də | di | -5.5 |
| 91703-00035-012622-0082965.npy | אויפן | ɔfn | afn | -5.5 |
