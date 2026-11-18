# Warm-up DOT: dynamic layer order with a layer warm-up (idea sketch)

Status: idea only, nothing run. Register entry D2; the full discussion and the B-DOT evidence are in `DIRECTION2_dynamic_order_revisited.md`.

## The idea

Let each token choose its own path through the layers: which layer runs next, in which order, and when to stop. Before any routing, warm up the layers so that they are used to running at many positions and depths. Then train a router on top of the warmed-up layers.

## How it works

The input and output layers are fixed and shared by every path; only the middle layers change. The model grows from one middle layer to many.

1. **Start with a 3-layer model.** Train input layer → middle layer 1 → output layer.
2. **Swap the middle layer.** Keep the input and output layers, put a new middle layer in place of the first one, and train it. Repeat until there are as many middle layers as the target model needs (for example 10 for a 12-layer model).
3. **Grow the depth.** Train paths with 2 middle layers, then 3, then 4, and so on up to the full depth.
4. **Vary the order at each depth.** For each number of middle layers, train different orders of the trained layers, for example:
    - 2 middle layers: input → 1 → 2 → output, input → 3 → 1 → output, …
    - 3 middle layers: input → 1 → 2 → 5 → output, input → 4 → 2 → 1 → output, …

    Not every combination is needed. The aim is only that each middle layer gets used to running at different positions and to taking the output of many different layers as its input, so a set of orders that covers this is sufficient.
5. **Route.** Train a router that picks each token's next middle layer, or STOP.
6. **Tune the warm-up for the router (optional).** Adjust steps 1–4 so that the router finds better paths.

Steps 1–4 are the warm-up; step 5 is the routing.

## Two variants

- **A: from scratch.** The procedure above on a new model.
- **B: on a pretrained model,** for example Qwen3: keep its input and output layers and apply steps 3–5 to its already-trained middle layers (to be confirmed).

## Why it might work

- Forcing random orders from the first step, when the model can do nothing yet, may hurt. Warming up the layers first may hurt less.
- Trained layers adapt to new inputs when training continues: stacked models such as SOLAR and G_stack work.
- A router that reads the context (for example the start of the sequence) rather than only the current state can keep paths stable within a context: code takes one path, prose another. Each layer then serves fewer kinds of paths and can specialize.
- The worry that interchangeable layers become alike may not apply, because the warm-up is short.

## First test, if resumed

At B-DOT scale (344M parameters, 2.5B tokens), with the warm-up's compute counted in the budget:

- **W-R:** warm-up, then random routing. Does the warm-up change the end point?
- **W-L:** warm-up, then a learned router, optionally reading the context.

Success, fixed in advance: W-L is at most 0.01 nats above the dense model at equal compute, and at least 0.01 nats below W-R.

## Design choices to settle

- What the input and output layers contain: only the embedding and the LM head, or also a fixed first and last transformer block.
- Whether the input and output layers stay frozen after the 3-layer model, or keep training with every new path.
- Whether already-trained middle layers stay frozen while deeper paths are trained, or all layers keep training.
- How many orders are enough at each depth, and which ones: for example a balanced subset in which every middle layer appears at each position and after many different layers.
- Paths trained one after another, as above, or several paths trained jointly on the same batch (FairNAS style), optionally weighted toward shorter paths.
- How the compute is split between the warm-up and the routing.
- Which paths the router may choose from: every order, or only paths close to a default order (a routing window).
- What the router reads: the current state only, or also a summary of the sequence; per token or once per segment.
- STOP is left out of the first test and added only if the order routing passes.
