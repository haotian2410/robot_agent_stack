# Skills and routes

The control package keeps the original skill implementations under its namespaced
package. Route A builds a scene and interaction registry from the final MuJoCo
model; Route B consumes an authored interaction sidecar when present, otherwise
uses the existing visual-grounding pipeline. Explicit semantic anchors fail with
`ANCHOR_NOT_FOUND`; unresolved search requests fail with `PERCEPTION_REQUIRED`.
