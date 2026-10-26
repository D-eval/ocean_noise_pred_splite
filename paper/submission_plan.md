# Submission plan (27 October 2026)

## Recommended venue

**OCEANS 2027 Aberdeen**, standard track **Underwater Acoustics and Acoustical Oceanography → Bioacoustics / Ocean noise**. The conference also lists **Sonar Signal/Image Processing and Communication → Sonar signal processing**. The ocean-specific data construction and the new-site whale-call check fit the first track best. See the [official technical tracks](https://aberdeen27.oceansconference.org/technical-tracks-2027/).

The [official call](https://aberdeen27.oceansconference.org/2027-call-for-abstracts/) lists **15 January 2027** for abstracts and **24 May 2027** for final papers. Its 2027 final-paper page limit is not yet stated on the pages checked. The [2026 OCEANS Monterey author instructions](https://monterey26.oceansconference.org/author-instructions/) specified 4–6 technical pages plus references; use that only as a provisional layout target until Aberdeen publishes its own instructions.

ICASSP 2027 is unsuitable for this manuscript cycle: its [full-paper deadline](https://2027.ieeeicassp.org/call-for-papers/) was 23 September 2026, and the [regular format](https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/) allows four technical pages plus an optional references-only page. [IEEE CAOS 2027](https://www.ieee-caos.org/call-for-paper/) is a possible alternative if its scope and fees are acceptable: six pages, up to eight with extra-page charges, with a 1 March 2027 full-paper deadline.

## Six-page conference cut order

The current English working draft is 10 pages including references (Chinese: 9 pages). The short 30-band discussion has already replaced its longer audit, while the three-seed and independent-site results remain intact. A venue-formatted six-page version still needs a separate cut after Aberdeen publishes its 2027 author instructions.

Preserve this argument in the main text:

1. Failed coarse condition → **information ladder** (one compact figure and its limits).
2. **128-group time-resolved residual interface** (architecture figure and causal data construction).
3. **BSR / Zero / Shuffled / Persistence / Forecast**, with the single-seed Oracle marked as an upper bound.
4. **Three separator seeds**: Persistence–Zero and Persistence–Shuffled remain positive in every seed-day cell; GRU–Persistence changes sign.
5. **Untouched GSL site**: report background-event response together with event retention and off-event alteration, without claiming isolated-stem recovery.

Cut first: old 30-band FiLM architecture details, its per-SNR plot, and the full-waveform copy-check table. Keep one short paragraph establishing why the information ladder was needed. Next, move the repeated 103-call target-calibration and weak-supervision trajectories out of the conference main text; summarize their tradeoffs in a limitation paragraph and retain detailed results in the repository or a supplement. Do not drop the three-seed or new-site evidence to make room for those exploratory studies.

## Submission gate

Before uploading, verify Aberdeen's own 2027 author kit, final page limit, anonymization rule, abstract field limit, and code-link policy. Replace the anonymous author block with the actual authors and affiliations only after the team approves the final manuscript. The present PDFs are working drafts, not a submission upload.
