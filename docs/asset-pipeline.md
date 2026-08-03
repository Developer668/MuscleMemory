# Asset Generation And Cache

The asset backend treats appearance and physics as separate products. Reference images and
TRELLIS GLB or OBJ outputs are tagged `rendering_only`. They can be loaded by a renderer, but
the collision schema cannot contain a visual-asset field. Physics receives either a
deterministic primitive or a separately generated convex hull with a recorded approval.

## Provider states

`ReferenceImageHttpAdapter` and `TrellisHttpAdapter` report one of four states:

- `unconfigured`: no endpoint exists, so the pipeline immediately uses the verified cache.
- `configured`: an endpoint exists but has not returned a validated artifact.
- `healthy`: the last request returned an artifact that passed media/container validation.
- `degraded`: the last request failed, timed out, or returned invalid bytes.

Both adapters use bounded timeouts. The hard upper bound is 30 seconds. Provider state is not
deployment proof and does not claim that an external generation happened when a fixture or
cache was used.

The provider-neutral request contract is an HTTP `POST` containing JSON. The reference-image
endpoint receives `prompt`, `request_id`, and `response_format`. TRELLIS receives
`image_base64`, `image_media_type`, and `output_format`. Both respond with:

```json
{
  "data_base64": "...",
  "media_type": "image/png",
  "format": "glb"
}
```

`format` is required only for mesh output. API keys are optional bearer tokens and must come
from a secret manager or runtime environment. The service-specific variable names are in
`config/services/asset-generation.env.example`.

## Content-addressed cache

`ContentAddressedAssetCache` stores blobs by their SHA-256 digest. A bundle manifest binds the
reference image, rendering mesh, request digest, provider labels, and truthful live/fallback
flags. The manifest, request alias, and each blob are verified on read. A changed blob,
manifest, checksum, or immutable request alias fails closed rather than falling through to an
unverified artifact.

The built-in fallback is a small deterministic appearance bundle. Pipeline startup seeds it
idempotently and reads it back through the same integrity checks. It is marked
`verified_fallback=true` and `live_generation=false`. Seed or verify it operationally with:

```bash
uv run python -m ops.assets.seed_verified_fallback --cache-dir artifacts/assets/cache
```

The fallback protects demo completion from network failure. It does not imply that a provider
ran, and it does not supply physical properties.

## Blocking physics approval

Dimensions, mass, friction, movable/static status, and collider choice are safety-critical.
Every agent-origin proposal creates an immutable `AssetApprovalRequirement`; any proposal that
marks one of those fields uncertain does the same. Until an authenticated human decision is
recorded, the result has `blocked_approval` state and contains no `WorldAdmissibleAsset`.

The approval ledger stores the exact proposal digest and one append-only human verdict. A
rejected proposal stays out of world assembly. An approved proposal produces a world-admissible
record with independent visual and collider identities. World assembly should accept only
`WorldAdmissibleAsset`, then still apply the normal world validation gate and category-specific
physical limits.
