## MODIFIED Requirements

### Requirement: Skewed matrix scheduling
For logical A[M,K] and B[K,N], the array SHALL accept A[m,k] at row m and
B[k,n] at column n during active step m+k and n+k respectively. Register
forwarding SHALL align the corresponding operands at PE[m,n]. After the final
skew step, one additional enabled step with invalid edge operands SHALL drain
the registered PE products before accumulator results are final.

#### Scenario: Rectangular logical matrix
- **WHEN** M and N do not fill every physical row and column and the final product-drain step completes
- **THEN** each active coordinate contains the exact Phase 0 matrix result and every unused accumulator remains zero

#### Scenario: Stalled active step
- **WHEN** enable is deasserted during skew or product drain
- **THEN** the schedule and queued products resume without loss or duplication on the next enabled step
