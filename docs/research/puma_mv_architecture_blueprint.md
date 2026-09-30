# PUMA-MV: Partial-label and Uncertainty-aware Multi-view Architecture

Status: research blueprint, 2026-09-29. The name is provisional. SOTA is a
target that must be established experimentally, not a claim made by this
document.

Implementation checkpoint:

- propensity-aware non-negative PU heatmap baseline;
- shared ResNet-18 FPN with true P3/P4/P5 spatial scales;
- differentiable metric multi-height, multi-scale sampler;
- permutation-invariant uncertainty-gated view/height/scale fusion with null token;
- dense high-recall detector retaining the legacy trainer interface;
- top-K local proposal initialization and continuous cylindrical query
  refinement from intact per-view feature maps;
- Hungarian matching with a partial-label PU query objective: observed queries
  receive metric coordinate regression while unmatched queries remain an
  unlabeled mixture instead of unconditional no-object targets;
- differentiable query-to-BEV rendering, two-GPU data parallelism, and AMP;
- positive-only full-view-to-camera-drop consistency with a no-grad teacher;
- metric local query separation and validation-controlled metric NMS;
- unit tests for full-label equivalence, empty-positive safety, projection,
  invalid-view masking, camera permutation invariance, query gradients, and
  end-to-end trainer output shapes.
- deterministic SCAR and visibility/scale-biased SAR split generation, with
  exhaustive test annotations enforced independently of the training split.
- a query-level SAR selection head trained through
  `P(s=1|x)=P(y=1|x)e(x)`, rate-anchored to mitigate the occupancy/propensity
  identifiability degeneracy.
- a dense-only warm-up followed by a five-epoch query-loss ramp, plus robust
  Student-t metric NLL, preventing random initial queries from overwhelming
  the pixel-averaged dense objective while still calibrating covariance.
- 64 sparse queries by default. The first real Wildtrack frame contains 38
  people, so the previous 32-query setting imposed a hard sparse-branch recall
  ceiling before learning.
- atomic full-state checkpoints (model, optimizer, scheduler, AMP scaler,
  history, and all RNG states), with exact continuation rather than silently
  restarting the learning-rate schedule.

Current verification: 43 unit/integration tests pass. A real seven-camera
Wildtrack frame (`7 x 3 x 720 x 1280`) completes an AMP forward/backward step
on an RTX 4050 6 GB at 4898 MiB peak allocated memory. All seven cameras have
valid projected samples and the result is finite. With 64 queries and the
robust coordinate objective, the random-initialization query loss was
approximately 8--11 in two smoke runs, versus about 257 with the previous
Gaussian NLL/32-query check. The training CLI applies a 0.02 query weight only
after a three-epoch dense warm-up and then ramps it over five epochs; this is
an integration/stability result, not an accuracy result.

The SCAR objectives use the single-training-set propensity identity
`E[l_-] + E[S/c (l_+ - l_-)]`: the negative marginal is evaluated over all
cells/queries, then the observed-positive contribution is subtracted after
inverse-propensity correction. Evaluating the marginal only on unmatched
samples would condition on `S=0` and bias the mixture; closed-form regression
tests now protect both the dense and query implementations against that bug.
The query-positive prior is allowed to rise to 0.95 rather than inheriting the
dense pixel cap of 0.5: a crowded Wildtrack frame can contain more than 32
people among 64 high-recall proposals, so the old cap would systematically
down-weight positives exactly in the crowded regime of interest.

`run_puma_dev.sh` is a deliberately non-reportable development gate: it uses
ten Wildtrack samples for training and the next ten for validation at
360x640, while exercising the complete query, camera-drop consistency,
checkpoint, score-cache, and evaluator paths. Official experiments retain
720x1280 and the declared 0--80/80--90/90--100 split protocol.

Not yet established: Wildtrack/MultiviewX MODA, gains over baselines, SAR
propensity learning, camera-drop consistency gains, and temporal memory. The
absence of these results prohibits an SOTA claim.

Memory profile (synthetic Wildtrack tensor geometry, batch 1, AMP backward,
RTX 4050 6 GB): 32/64 feature/fusion channels peaked at 6372 MiB allocated;
16/48 peaked at about 4800 MiB. Sampling feature and uncertainty jointly cut
the 16/48 sequential step from about 99 to 12.8 seconds. Parallel view encoding
then reduced it to 10.5 seconds at 4784 MiB. The CLI defaults to sequential
16/48 for 6 GB safety, while `run_puma_grid.sh` uses parallel 32/64 on 16 GB
Kaggle T4 GPUs. These figures measure feasibility, not detection accuracy;
the T4 speedup still needs to be measured directly.

The legacy evaluator used a fixed 20-cell (0.5 m) point-NMS radius, which can
erase a valid neighbour after the query decoder has separated it. PUMA exposes
NMS in metres and defaults to 0.3 m; this hyperparameter must be frozen from
validation, while the official evaluation matching radius remains unchanged.

Evaluation hardening implemented after audit:

- batch-size greater than one now preserves the batch/frame dimension when
  exporting detections;
- frames with zero predictions remain in the evaluator and contribute their
  ground-truth people as false negatives;
- the standard final split is evaluated only once by benchmark scripts;
- `run_puma_validation.sh` uses 0--80% for training and 80--90% for validation,
  caches raw BEV maps once, and sweeps threshold/NMS offline on that same
  checkpoint; final runs retrain on 0--90% and evaluate 90--100%.

## 1. Research question

Can a multi-view pedestrian detector retain the localization accuracy of
modern geometry-aware/query-based models while remaining statistically valid
when only a fraction of pedestrians are annotated?

The central failure in partially annotated training is not merely a shortage
of positive examples. Every omitted pedestrian is incorrectly optimized as
background. PUMA-MV therefore treats an absent label as *unlabeled evidence*,
not a negative label, and couples this treatment to multi-view geometry.

## 2. Evidence from the current literature

### 2.1 Multi-view pedestrian detection

| Work | Main contribution | Reported strength | Limitation relevant here |
|---|---|---|---|
| [MVDet, ECCV 2020](https://arxiv.org/abs/2007.07247) | Project per-view CNN features to a ground-plane BEV and concatenate them | 88.2 MODA on Wildtrack | Ground-plane warp distorts upper-body features; fixed camera count/order; dense heatmap and complete-label assumption |
| [SHOT, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Song_Stacked_Homography_Transformations_for_Multi-View_Pedestrian_Detection_ICCV_2021_paper.html) | Stack homographies at multiple human heights and learn soft height selection | Better 3D correspondence than one ground-plane homography | Dense lifting remains scene/calibration dependent and is trained with exhaustive negatives |
| [MVDeTr, MM 2021](https://github.com/hou-yz/MVDeTr) | Deformable attention corrects projected BEV features; view-coherent augmentation | 91.5/93.7 MODA on Wildtrack/MultiviewX | Attention operates after a distorted projection; no missing-label model |
| [3DROM, ECCV 2022](https://github.com/xjtlu-cvlab/3DROM) | Multi-layer projection and geometry-consistent 3D occlusion augmentation | Strong heavy-occlusion performance | Expensive dense processing; no unknown-label state |
| [MVTT, WACV 2023](https://openaccess.thecvf.com/content/WACV2023W/RWS/papers/Lee_Multi-View_Target_Transformation_for_Pedestrian_Detection_WACVW_2023_paper.pdf) | Transform targets consistently across views | 94.1 MODA reported on Wildtrack | Target design still presumes complete annotations |
| [Booster-SHOT, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Hwang_Booster-SHOT_Boosting_Stacked_Homography_Transformations_for_Multiview_Pedestrian_Detection_With_WACV_2024_paper.pdf) | Channel/spatial attention over SHOT features | 92.9/94.2 MODA | Attention does not correct false-negative supervision |
| [MVFP, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/papers/Aung_Enhancing_Multi-View_Pedestrian_Detection_Through_Generalized_3D_Feature_Pulling_WACV_2024_paper.pdf) | Pull intact image features into a 3D volume and max-fuse views | Strong standard and cross-scene accuracy | Full volumes are expensive; max fusion discards uncertainty and view multiplicity |
| [EarlyBird, WACV 2024](https://arxiv.org/abs/2310.13350) | Early BEV fusion and learned re-identification | +4.6 MOTA and +5.6 IDF1 over its tracking baseline | Tracking improvement does not address partial detection labels |
| [TrackTacular, CVPRW 2024](https://arxiv.org/abs/2403.12573) | Modern lifting plus temporal aggregation for detection/tracking | 92.1/96.5 MODA on Wildtrack/MultiviewX | Average/dense aggregation and ordinary negative supervision remain vulnerable to missing labels |
| [CaMuViD, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Daryani_CaMuViD_Calibration-Free_Multi-View_Detection_CVPR_2025_paper.pdf) | Learned image/shared-space projection without supplied calibration | 95.0/96.5 MODA and strong cross-dataset results | Learned mapping can still overfit appearance/configuration; no partial-label objective |
| [Probabilistic Occupancy Volume, CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/WiCV/html/Alturki_Enhanced_Multi-View_Pedestrian_Detection_Using_Probabilistic_Occupancy_Volume_CVPRW_2025_paper.html) | Visual-hull occupancy guides 3D feature pulling | 97.3 MODA on MultiviewX | Foreground masks/background modeling can be brittle; 93.6 MODA on Wildtrack; no partial-label treatment |
| [DCHM, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ma_DCHM_Depth-Consistent_Human_Modeling_for_Multiview_Detection_ICCV_2025_paper.html) | Multi-view-consistent monocular depth and point-cloud human modeling | Raises its UMPD baseline from 76.6 to 84.2 MODA | Reconstruction is costly and still below top supervised detectors |
| [MSMVD, BMVC 2025](https://arxiv.org/abs/2508.20447) | Project multi-scale image features into matching multi-scale BEV features | +4.5 MODA over prior GMVD best | Primarily solves scale variation, not missing labels or label bias |
| [MVDGC, 2026](https://arxiv.org/abs/2607.00273) | Sparse 3D cylindrical queries jointly predict BEV points and image boxes | 94.2 MODA, 83.1 MODP on Wildtrack; strong localization | Hungarian/no-object supervision still assumes an exhaustive object set; per-view box constraints disappear with a missing person |
| [MV2GF, 2026](https://arxiv.org/abs/2608.20639) | Visual-geometric foundation features and predicted pointmaps | Promising unseen-camera generalization | Foundation-model cost and complete-label training remain; results need independent replication |

Reported scores are not directly comparable unless splits, image backbone,
augmentation, thresholding, and NMS protocol are identical.

### 2.2 Partial and missing annotations

| Work | Transferable idea |
|---|---|
| [Soft Sampling](https://arxiv.org/abs/1806.06986) | Reduce uncertain-background gradients instead of deleting all background. At 50% dropped VOC labels it reports 73.50 mAP versus 69.11 for the baseline and 74.54 for the upper bound. |
| [Object Detection as a Positive-Unlabeled Problem](https://arxiv.org/abs/2002.04672) | Replace the assumption “unmatched means background” with a PU risk objective. |
| [BRL](https://arxiv.org/abs/2002.05274) | Recalibrate hard/confusing negative loss in a one-stage detector. |
| [Co-mining](https://arxiv.org/abs/2012.01950) | Cross-augmentation agreement can mine missing instances, but hard pseudo-label exchange can amplify errors. |
| [nnPU](https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html) | Clamp the estimated negative risk to prevent a flexible neural model from driving an unbiased PU estimator below zero. |
| [SAR-PU](https://arxiv.org/abs/1808.08755) | Model annotation propensity as a function of observed attributes when positives are not missing completely at random. |
| [SparseDet, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Suri_SparseDet_Improving_Sparsely_Annotated_Object_Detection_with_Pseudo-positive_Mining_ICCV_2023_paper.pdf) | Partition proposals into labeled, unlabeled, and reliable background; use consistency/self-supervision on unlabeled regions. It reports average gains of 2.6, 3.9, and 9.6 mAP as sparsity increases across three splits. |
| [Calibrated Teacher, AAAI 2023](https://ojs.aaai.org/index.php/AAAI/article/view/25349) | Confidence calibration and false-negative-aware weighting are essential if a teacher is used. |

## 3. Unresolved intersection

The two literatures largely solve different halves of the problem:

1. Modern MVPD assumes that every unannotated BEV location is background.
2. Partial-label detection usually operates on single-view boxes and does not
   exploit calibrated N-camera evidence.
3. Existing YOLO pseudo-label pipelines import a dataset-dependent teacher,
   foot-point error, threshold sensitivity, and confirmation bias.
4. Dense BEV heatmaps preserve recall but merge close pedestrians and quantize
   coordinates.
5. Sparse query models localize precisely but their unmatched-query
   no-object loss is invalid with missing annotations and can suppress the very
   people the model should recover.

PUMA-MV is designed around this gap rather than around a stronger image
backbone alone.

## 4. Proposed architecture

### 4.1 Overview

```text
N synchronized views
        │
Shared multi-scale image encoder (P3/P4/P5)
        │
Geometry tokens: camera ray, scale, visibility, calibration uncertainty
        │
┌──────────────────────────────────────────────────────────────────┐
│ A. Dense high-recall multi-height BEV evidence                   │
│    project BEV samples at z={foot, knee, torso, head} into views │
│    permutation-invariant uncertainty-gated set attention          │
└────────────────────────────┬─────────────────────────────────────┘
                             │ top-K peaks + learned discovery queries
┌────────────────────────────▼─────────────────────────────────────┐
│ B. Sparse continuous 3D cylindrical query decoder               │
│    sample intact multi-scale image features in every visible view│
│    local query interaction separates close pedestrians           │
│    iterative (x,y,r,h,visibility,uncertainty) refinement          │
└───────────────┬───────────────────────────────┬──────────────────┘
                │                               │
        continuous BEV points          projected per-view boxes
                │                               │
        PU/set objective          dual geometric consistency
                └───────────────┬───────────────┘
                                │
                 optional visibility-gated temporal memory
```

### 4.2 Shared multi-scale encoder

Input is `B x N x 3 x H x W`. A shared encoder produces P3/P4/P5 image
features. The initial implementation should use a reproducible ConvNeXt-T or
ResNet-50 FPN; a frozen geometric foundation branch is a later ablation, not a
required dependency.

Multi-scale features address the failure observed by MSMVD: the same person
can be tiny in one camera and large in another. Camera identity is not encoded
as a fixed channel index. Each feature receives camera-ray and projection-scale
embeddings so fusion remains permutation invariant and accepts variable N.

### 4.3 Dense multi-height evidence proposal

For every reduced BEV cell `q=(x,y)`, define a small vertical sample set:

```text
z = [0.0, 0.45, 0.95, 1.45, 1.75] metres
```

Project these points into each visible camera and bilinearly sample P3/P4/P5.
This keeps the useful multi-plane principle of SHOT without creating a dense
3D feature volume. A set-attention block aggregates the `(view,height,scale)`
tokens:

```text
a_c,h,l = softmax(score(feature, ray, visibility, uncertainty))
F_bev(q) = sum a_c,h,l * value_c,h,l
```

Include a learned null token so an invalid or fully occluded set is not forced
to select a bad camera. Predict:

- dense high-recall objectness;
- center offset within the BEV cell;
- evidence uncertainty;
- expected visible-view count.

This stage is a proposal mechanism, not the final quantized detection head.

### 4.4 Sparse cylindrical refinement

Initialize queries from the top-K dense local maxima and add a small set of
learned discovery queries. A query stores `(x,y,r,h)` plus an embedding.

At every decoder layer:

1. Project foot, torso, head, and cylinder boundary samples into every camera.
2. Sample intact multi-scale image features using deformable attention.
3. Perform per-query cross-view set attention with visibility/uncertainty gates.
4. Perform radius-limited query self-attention in BEV to reason about nearby
   people without globally merging them.
5. Refine continuous `(x,y,r,h)`, objectness, covariance, per-view visibility,
   and per-view box residuals.

This retains MVDGC's object-centric geometric strength while the dense proposal
branch protects recall in crowded regions and under missing labels.

### 4.5 Optional temporal memory

After validating the single-frame model, add a visibility-gated query memory:

- high-confidence current queries may create tracks;
- low-confidence queries may continue an existing track but cannot create one;
- memory is retained when all current views are occluded;
- velocity predicts the next BEV reference point;
- training uses adjacent frames even when only one frame contains an observed
  annotation.

This imports the useful temporal principle of EarlyBird/TrackTacular without
letting temporal hallucinations inflate false positives.

## 5. Partial-label objective

### 5.1 Three-state supervision

Every location/query is assigned one of:

1. `observed_positive`: matched to a supplied BEV annotation;
2. `reliable_background`: supported as empty by geometry and calibrated model
   evidence;
3. `unlabeled`: may be a missing pedestrian and must not receive ordinary
   no-object loss.

Regression losses apply only to observed positives. Unknown unmatched queries
receive consistency and PU risk, not a hard background target.

### 5.2 Propensity-aware nnPU risk

Let `s=1` mean a positive was annotated and `y=1` mean a pedestrian truly
exists. For synthetic random drop, annotation propensity is known:

```text
e(x) = P(s=1 | y=1,x) = 1 - drop_ratio
```

For real missingness, learn a bounded propensity head from attributes that
affect annotation probability but are not the desired occupancy label:

- visible-camera count;
- projected person scale;
- crowd density;
- border/occlusion state;
- image quality.

Use inverse-propensity weighting for observed positives and a non-negative PU
risk for unlabeled candidates. Start with SCAR-nnPU on controlled dropped
annotations; enable SAR only after a SCAR-vs-SAR diagnostic.

### 5.3 Multi-view consistency without hard pseudo labels

For the same synchronized frame, run two geometry-valid perturbations:

- all available cameras with weak photometric augmentation;
- camera dropout plus strong per-view photometric augmentation.

Match queries by continuous BEV distance and minimize symmetric prediction
consistency only where predictive uncertainty is low. The stopped-gradient
target remains soft; it is never serialized as a hard pseudo annotation.

### 5.4 Dual geometric loss

For an observed pedestrian, project its refined cylinder into every camera in
which the person is visible. Optimize BEV point likelihood and image-box
alignment jointly. When the BEV label is missing, cross-view agreement still
constrains shape and location, but it does not by itself declare a positive.

### 5.5 Total loss

```text
L = L_observed_set
  + lambda_pu   * L_propensity_nnPU
  + lambda_geo  * L_dual_geometry
  + lambda_cons * L_camera_dropout_consistency
  + lambda_sep  * L_local_query_separation
  + lambda_temp * L_temporal_consistency       # optional second phase
  + lambda_cal  * L_uncertainty_calibration
```

The initial safe schedule is to warm up on observed positives and reliable
background, then ramp `lambda_pu` and `lambda_cons` over 3-5 epochs. This avoids
early predictions defining the unknown-region dynamics.

## 6. Why this is not just another pseudo-label method

- It never claims that an external detector prediction is GT.
- Missing locations remain a mixture of positive and negative in the risk.
- Cross-view/camera-drop outputs are soft consistency constraints.
- Annotation selection bias is explicitly modeled through propensity.
- Geometry is part of the detector and objective, not an offline label
  generator.

An optional calibrated external teacher can be evaluated later, but it is not
part of the core architecture.

## 7. Failure modes addressed

| Existing failure | PUMA-MV mechanism |
|---|---|
| Missing person trained as background | Three-state assignment plus nnPU |
| Random-drop assumption fails on occluded people | Visibility/density-conditioned SAR propensity |
| One-plane homography stretches bodies | Multi-height projection and intact-image query sampling |
| Dense heatmaps merge nearby people | Continuous local cylindrical queries and separation loss |
| Sparse queries miss weak people | Dense high-recall proposals plus discovery queries |
| Fixed camera count/order | Permutation-invariant set attention and camera dropout |
| Bad camera dominates fusion | Visibility and uncertainty gating with null token |
| Calibration/projective error | Continuous offset/covariance and deformable sampling |
| Person scale differs strongly across views | Multi-scale sampling before fusion |
| Single-frame total occlusion | Optional visibility-gated temporal query memory |
| YOLO domain/threshold dependence | No external detector in the core method |

## 8. Claims that must be tested, not assumed

1. nnPU on dense BEV cells may be unstable because occupancy is extremely
   sparse and spatially correlated. Candidate sampling and prior estimation
   require ablation.
2. Propensity and occupancy are not identifiable without assumptions. The
   synthetic random-drop protocol validates SCAR; real missingness needs SAR
   sensitivity analysis.
3. A hybrid dense/sparse model may increase memory and latency. Compare against
   a sparse-only and dense-only model at equal backbone/FLOPs.
4. Standard Wildtrack/MultiviewX train-test splits contain substantial scene
   overlap, so a high in-scene MODA alone is not evidence of generalization.
5. No SOTA claim is valid until results are reproduced with the official
   evaluation radius, full test annotations, identical NMS, and multiple seeds.

## 9. Experimental protocol

### 9.1 Benchmarks

- Wildtrack and MultiviewX, standard fully annotated protocol.
- Controlled instance drops at 20%, 45%, 60%, with full-label evaluation.
- Both SCAR drops and visibility-biased SAR drops.
- GMVD cross-scene and unseen camera configurations.
- Camera-drop tests at train and inference time.
- Calibration perturbation tests on intrinsics/extrinsics.

### 9.2 Required baselines

```text
MVDet + GaussianMSE
MVDet + BRL implementation
MVDet + YOLO pixel-wise-max pseudo loss
MVDet + nnPU only
MVDGC or closest reproducible sparse-query baseline
PUMA dense only
PUMA sparse only
PUMA dense+sparse
PUMA dense+sparse+PU
PUMA full consistency
PUMA full temporal
```

### 9.3 Metrics

- MODA, MODP, precision, recall, F1;
- FP and FN separately, stratified by crowd density and visible-view count;
- calibration ECE/Brier score;
- robustness area under the curve over label-drop rate;
- cross-scene/camera-drop degradation;
- parameters, FLOPs, peak memory, and FPS;
- mean and standard deviation over at least three seeds.

Threshold and NMS radius must be selected on validation only. Evaluation uses
the complete ground truth even when training annotations are dropped.

## 10. Implementation sequence

### Gate A: trustworthy partial-label benchmark

1. Freeze one official evaluator and matching radius.
2. Generate deterministic SCAR and visibility-biased SAR splits with manifests.
3. Reproduce full/drop MVDet, BRL, and YOLO-pseudo baselines over three seeds.
4. Record FP/FN by density and number of visible cameras.

### Gate B: statistical baseline before architecture novelty

1. Implement sampled BEV nnPU loss.
2. Validate class-prior recovery on synthetic drops.
3. Add three-state masks and camera-drop consistency.
4. Require improvement over BRL at all drop rates without degrading the full
   annotation baseline by more than the predeclared tolerance.

### Gate C: geometry and fusion

1. Add shared FPN and multi-height differentiable sampler.
2. Add permutation-invariant uncertainty-gated view fusion.
3. Compare ground-plane, SHOT-like, and proposed lifting at equal compute.

### Gate D: hybrid continuous detector

1. Add dense candidate head.
2. Add cylindrical query refinement and continuous point output.
3. Add dual BEV/image geometric losses.
4. Add local query interaction/separation for crowded regions.

### Gate E: temporal and generalization

1. Add camera-drop training and variable-N evaluation.
2. Add temporal query memory.
3. Evaluate GMVD and calibration perturbations.
4. Only after all gates pass, compare against published SOTA under matched
   protocols and make a defensible SOTA claim.

## 11. Minimum publishable contribution

The smallest coherent research contribution is not the complete temporal
system. It is:

> A hybrid dense-to-sparse, geometry-aware multi-view pedestrian detector with
> propensity-corrected positive-unlabeled set learning, showing robustness to
> both random and visibility-biased missing annotations.

The core ablation must prove independent gains from:

1. three-state/PU supervision;
2. permutation-invariant uncertainty-gated fusion;
3. dense-to-sparse cylindrical refinement.

If these do not independently improve controlled baselines, adding a larger
backbone or external pseudo-label teacher does not validate the hypothesis.
