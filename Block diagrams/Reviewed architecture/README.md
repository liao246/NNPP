# MNIST accelerator — diagram review copy

Open **MNIST accelerator — reviewed.drawio** in draw.io / diagrams.net. All eight pages are editable vector shapes, grouped text, and attached connectors. Your original file is unchanged.

Open **Preview.html** for a zoomable page selector. The `previews` folder contains individual SVG and PNG pages. The PNGs are exported from the editable draw.io file itself.

## Pages

1. System datapath — both FC engines and registered activation handoffs.
2. Top control — controller structure, input acceptance, cycle counting, and output capture.
3. Top FSM — all outputs, transitions, wait loops, and held completion.
4. FC layer — controller, explicit counter next-value logic, addresses, and output capture.
5. FC FSM — two fill states and one capture state per batch.
6. MAC lane — local weight/bias memories and signed accumulator datapath.
7. Requant + ReLU — newly specified one-stage module with output hold and aligned valid.
8. Timing & mapping — a review reference for cycle accounting, widths, and lane ownership.

## Preserved architecture

196 → 64 → 10; two FC engines; 16 lanes per engine; per-lane weight and bias memories; contiguous lane ownership; the weight-address buffer and activation register inside each MAC; FC2's external delay register; a 64-wide requant bank; the separate 512-bit activation capture bank; raw signed FC2 scores into clocked argmax; registered prediction and correctness outputs.

## Corrections incorporated

- Capture index matches contiguous ownership: neuron = lane × NUM_BATCHES + batch. The validity guard uses that same index.
- Added FILL_0 and FILL_1 to the FC FSM. The counter advances while MAC updates are disabled, then exactly N_IN products accumulate. Seven states still fit three bits.
- Weight addresses use the clamped input index. Counter terminal comparisons and hold behavior are explicit.
- Batch counter initializes in IDLE and advances only after nonfinal captures. It is not cleared between batches.
- Single-batch address widths are guarded to at least one bit. Inactive FC2 lane memory entries are zero initialized.
- The activation bank captures only on requant_valid and otherwise holds. The newly drawn requant module also holds its output; the separate bank is retained to preserve your architecture.
- Added FC2's activation-address return path to the activation selector.
- Image selection and cycle-counter clearing use accept_start, qualified by IDLE or DONE. Image indices above 19 clamp to 19.
- The label comparator uses the valid argmax result before the final digit register. Prediction and correctness capture together as the top FSM enters DONE; the old delayed-enable FF is removed.
- Top DONE restarts directly into FC1_START. Top completion is held; layer completion remains a one-cycle pulse.
- Clocked modules, signed widths, register feedback, output defaults, and debug bit ordering are explicit.

## Reading conventions

Blue = datapath logic/module; purple = register; green = memory; orange = control; gray = explanatory note. A triangle denotes the common clock. All sequential blocks use the same clk and synchronous reset; reset wires are intentionally omitted. `s32` means a signed 32-bit value.

Named control inputs printed above blocks and named stubs connect by identical signal name across pages; these avoid long crossing wires and do not add storage or logic. State diagrams specify the controller logic inside their matching datapath pages. Main datapath pages use hierarchical module boxes; internal next-value details are shown on the corresponding detail pages.

## Deliberately unspecified by the original diagrams

The internal argmax implementation and exact latency remain open. The diagrams require a clocked result with aligned valid, signed comparisons, and smallest-index tie breaking. They do not invent a new comparator-tree or scan architecture. The top controller waits for valid and works with a latency of at least one cycle.

SHIFT1 remains a calibrated design constant; these diagrams do not take its value from old RTL. For SHIFT1=0, the rounding offset is zero. The signed 32-bit rounding sum must stay in range for the exported model.

These are proposed corrected design diagrams, not synthesized or simulated RTL. Verification here covers diagram rendering, connectivity/layout checks, and a cycle-level model of the stated FC control schedule. No existing RTL was used to define module behavior or edited.
