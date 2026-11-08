#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0, d1) -> (0, d1)>
#map5 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @forward(%arg0: tensor<2x8xf32>) -> tensor<2x4xf32> {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0xFF800000 : f32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense_resource<torch_tensor_4_torch.float32_1> : tensor<4xf32>
    %cst_2 = arith.constant dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>
    %cst_3 = arith.constant 1.000000e-08 : f64
    %cst_4 = arith.constant 1.270000e+02 : f32
    %cst_5 = arith.constant -1.270000e+02 : f32
    %cst_6 = arith.constant dense_resource<torch_tensor_8_4_torch.int8> : tensor<8x4xi8>
    %0 = tensor.empty() : tensor<2x8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<2x8xf32>) outs(%0 : tensor<2x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = math.absf %in : f32
      linalg.yield %23 : f32
    } -> tensor<2x8xf32>
    %2 = tensor.empty() : tensor<2xi64>
    %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<2xi64>) -> tensor<2xi64>
    %4 = tensor.empty() : tensor<2xf32>
    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<2xf32>) -> tensor<2xf32>
    %6:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%1 : tensor<2x8xf32>) outs(%5, %3 : tensor<2xf32>, tensor<2xi64>) {
    ^bb0(%in: f32, %out: f32, %out_8: i64):
      %23 = linalg.index 1 : index
      %24 = arith.index_cast %23 : index to i64
      %25 = arith.maximumf %in, %out : f32
      %26 = arith.cmpf ogt, %in, %out : f32
      %27 = arith.select %26, %24, %out_8 : i64
      linalg.yield %25, %27 : f32, i64
    } -> (tensor<2xf32>, tensor<2xi64>)
    %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%6#0 : tensor<2xf32>) outs(%4 : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.truncf %cst_3 : f64 to f32
      %24 = arith.cmpf ult, %in, %23 : f32
      %25 = arith.select %24, %23, %in : f32
      linalg.yield %25 : f32
    } -> tensor<2xf32>
    %8 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%7 : tensor<2xf32>) outs(%4 : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.divf %in, %cst_4 : f32
      linalg.yield %23 : f32
    } -> tensor<2xf32>
    %expanded = tensor.expand_shape %8 [[0, 1]] output_shape [2, 1] : tensor<2xf32> into tensor<2x1xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %expanded : tensor<2x8xf32>, tensor<2x1xf32>) outs(%0 : tensor<2x8xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.divf %in, %in_8 : f32
      linalg.yield %23 : f32
    } -> tensor<2x8xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<2x8xf32>) outs(%0 : tensor<2x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = math.roundeven %in : f32
      linalg.yield %23 : f32
    } -> tensor<2x8xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<2x8xf32>) outs(%0 : tensor<2x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.cmpf ult, %in, %cst_5 : f32
      %24 = arith.select %23, %cst_5, %in : f32
      %25 = arith.cmpf ugt, %24, %cst_4 : f32
      %26 = arith.select %25, %cst_4, %24 : f32
      linalg.yield %26 : f32
    } -> tensor<2x8xf32>
    %12 = tensor.empty() : tensor<2x8xi8>
    %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<2x8xf32>) outs(%12 : tensor<2x8xi8>) {
    ^bb0(%in: f32, %out: i8):
      %23 = arith.fptosi %in : f32 to i8
      linalg.yield %23 : i8
    } -> tensor<2x8xi8>
    %14 = tensor.empty() : tensor<2x4xi32>
    %15 = linalg.fill ins(%c0_i32 : i32) outs(%14 : tensor<2x4xi32>) -> tensor<2x4xi32>
    %16 = linalg.matmul ins(%13, %cst_6 : tensor<2x8xi8>, tensor<8x4xi8>) outs(%15 : tensor<2x4xi32>) -> tensor<2x4xi32>
    %17 = tensor.empty() : tensor<2x4xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<2x4xi32>) outs(%17 : tensor<2x4xf32>) {
    ^bb0(%in: i32, %out: f32):
      %23 = arith.sitofp %in : i32 to f32
      linalg.yield %23 : f32
    } -> tensor<2x4xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %expanded : tensor<2x4xf32>, tensor<2x1xf32>) outs(%17 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      linalg.yield %23 : f32
    } -> tensor<2x4xf32>
    %expanded_7 = tensor.expand_shape %cst_2 [[0, 1]] output_shape [1, 4] : tensor<4xf32> into tensor<1x4xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %expanded_7 : tensor<2x4xf32>, tensor<1x4xf32>) outs(%17 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      linalg.yield %23 : f32
    } -> tensor<2x4xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map5, #map], iterator_types = ["parallel", "parallel"]} ins(%20, %cst_1 : tensor<2x4xf32>, tensor<4xf32>) outs(%17 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.addf %in, %in_8 : f32
      linalg.yield %23 : f32
    } -> tensor<2x4xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<2x4xf32>) outs(%17 : tensor<2x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %23 = arith.cmpf ugt, %in, %cst_0 : f32
      %24 = arith.select %23, %in, %cst_0 : f32
      linalg.yield %24 : f32
    } -> tensor<2x4xf32>
    return %22 : tensor<2x4xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_4_torch.float32_1: "0x0400000086AA6ABE86A826BE53EF7CBE2089A9BE",
      torch_tensor_4_torch.float32: "0x040000000329163B5B4C2E3BF77F173B9F3F293B",
      torch_tensor_8_4_torch.int8: "0x01000000FFF43CEA53235C0E81D8987C8EE6BD81C58138AA29A87FDDFDC9E1CB7A057276"
    }
  }
#-}

