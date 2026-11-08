# Registering a middle-end pass

`src/export/middle-end/torch/transforms/` holds `pynqo-middle-end`'s rewrites on
the torch dialect. They run before torch-mlir lowers the module to linalg,
while layers are still recognisable. Like the quantization library, they know
nothing about any accelerator.

## Rules

- Do not mention a device in names, comments or pass arguments: no `npu`, no
  vendor path. Pass arguments start with `opt-`.
- One pass per file, named for what it rewrites: `DecomposeIntMm.cpp`.
- A pass changes nothing when its pattern is absent, and keeps the values it
  replaces numerically identical.
- A pass that fills a torch-mlir gap says so in its header, with the upstream
  change that makes it unnecessary, so it can be removed after an upgrade.

## Adding a pass

1. Write the pattern and the pass in `torch/transforms/<Name>.cpp`:

   ```cpp
   namespace {

   // One rewrite: match one op, build its replacement.
   struct MyRewrite : OpRewritePattern<AtenFooOp> {
     using OpRewritePattern::OpRewritePattern;

     LogicalResult matchAndRewrite(AtenFooOp op, PatternRewriter &rewriter) const override {
       if (/* the op is not one this rewrite handles */)
         return failure();
       rewriter.replaceOpWithNewOp<AtenBarOp>(op, op.getType(), op.getSelf());
       return success();
     }
   };

   // The pass: apply the pattern to the whole module until nothing matches.
   struct MyRewritePass : PassWrapper<MyRewritePass, OperationPass<ModuleOp>> {
     MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyRewritePass)

     StringRef getArgument() const final { return "opt-my-rewrite"; }

     void runOnOperation() override {
       RewritePatternSet patterns(&getContext());
       patterns.add<MyRewrite>(&getContext());
       if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
         signalPassFailure();
     }
   };

   } // namespace

   // The file's only public symbol.
   std::unique_ptr<Pass> middle_end::createMyRewritePass() {
     return std::make_unique<MyRewritePass>();
   }
   ```

2. Declare `createMyRewritePass()` in `torch/transforms/Passes.h`, with a
   one-line comment saying what it rewrites.
3. Add it to the pass manager in `pynqo_middle_end.cpp`, before the torch-mlir
   pipelines. Order matters: a pass that matches an op must run before the
   pass that replaces it.
4. Add the `.cpp` to `add_llvm_executable(pynqo-middle-end ...)` in
   `src/export/middle-end/CMakeLists.txt`.
5. Test it in `src/test/tests/test_export_middle_end.py`. Run with
   `--emit=torch` to check the rewrite itself, and with the default
   `--emit=linalg` to check the module still lowers.

To share a pattern between passes, move it into a
`populate<Name>Patterns(RewritePatternSet &)` function and call that from each
pass.
