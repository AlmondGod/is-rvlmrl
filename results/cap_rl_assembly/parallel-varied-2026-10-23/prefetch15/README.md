# Training-only speculative evaluations

The two-worker prefetch attempt completed five of 28 planned entries in
2,528.11 seconds. One was already cached; four required fresh physics. Four
entries succeeded and one failed. These are training programs, not a selected
policy evaluation or an improvement result. The process stopped after the
learner was killed, finishing work already started.

The original metadata, programs, results and source are preserved in
`training-prefetch15-records.tar.gz` on the speed experiment release. The
verified archive has 21,258 bytes and SHA256
`f9fa1d200149dcf3518e2a265b465104d88f772dc5878c0caa115ae5d9eae409`.
Physical evidence is also retained in the online checkpoint archives.

Do not add its elapsed time to overlapping GPU allocation time. No separate
learner or optimizer ran here. A cached result can enter online learning only
when that learner independently requests the identical code and layout under
the same certified physical provenance.
