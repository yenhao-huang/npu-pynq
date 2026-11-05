module {
  func.func @forward(%arg0: !torch.vtensor<[2,8],f32>) -> !torch.vtensor<[2,4],f32> {
    %0 = torch.aten.abs %arg0 : !torch.vtensor<[2,8],f32> -> !torch.vtensor<[2,8],f32>
    %int-1 = torch.constant.int -1
    %1 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %2 = torch.aten.amax %0, %1, %false : !torch.vtensor<[2,8],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2],f32>
    %float1.000000e-08 = torch.constant.float 1.000000e-08
    %none = torch.constant.none
    %3 = torch.aten.clamp %2, %float1.000000e-08, %none : !torch.vtensor<[2],f32>, !torch.float, !torch.none -> !torch.vtensor<[2],f32>
    %int127 = torch.constant.int 127
    %4 = torch.aten.div.Scalar %3, %int127 : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[2],f32>
    %int1 = torch.constant.int 1
    %5 = torch.aten.unsqueeze %4, %int1 : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[2,1],f32>
    %6 = torch.aten.div.Tensor %arg0, %5 : !torch.vtensor<[2,8],f32>, !torch.vtensor<[2,1],f32> -> !torch.vtensor<[2,8],f32>
    %7 = torch.aten.round %6 : !torch.vtensor<[2,8],f32> -> !torch.vtensor<[2,8],f32>
    %int-127 = torch.constant.int -127
    %int127_0 = torch.constant.int 127
    %8 = torch.aten.clamp %7, %int-127, %int127_0 : !torch.vtensor<[2,8],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,8],f32>
    %none_1 = torch.constant.none
    %none_2 = torch.constant.none
    %int6 = torch.constant.int 6
    %cpu = torch.constant.device "cpu"
    %int0 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %8, %none_1, %none_2, %int6, %cpu, %int0 : !torch.vtensor<[2,8],f32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int1_3 = torch.constant.int 1
    %none_4 = torch.constant.none
    %none_5 = torch.constant.none
    %none_6 = torch.constant.none
    %false_7 = torch.constant.bool false
    %none_8 = torch.constant.none
    %9 = torch.aten._to_copy %8, %int1_3, %none_4, %none_5, %none_6, %false_7, %none_8 : !torch.vtensor<[2,8],f32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[2,8],si8>
    %10 = torch.vtensor.literal(dense_resource<torch_tensor_8_4_torch.int8> : tensor<8x4xsi8>) : !torch.vtensor<[8,4],si8>
    %11 = torch.aten._int_mm %9, %10 : !torch.vtensor<[2,8],si8>, !torch.vtensor<[8,4],si8> -> !torch.vtensor<[2,4],si32>
    %none_9 = torch.constant.none
    %none_10 = torch.constant.none
    %int3 = torch.constant.int 3
    %cpu_11 = torch.constant.device "cpu"
    %int0_12 = torch.constant.int 0
    torch.aten._assert_tensor_metadata %11, %none_9, %none_10, %int3, %cpu_11, %int0_12 : !torch.vtensor<[2,4],si32>, !torch.none, !torch.none, !torch.int, !torch.Device, !torch.int
    %int6_13 = torch.constant.int 6
    %none_14 = torch.constant.none
    %none_15 = torch.constant.none
    %none_16 = torch.constant.none
    %false_17 = torch.constant.bool false
    %none_18 = torch.constant.none
    %12 = torch.aten._to_copy %11, %int6_13, %none_14, %none_15, %none_16, %false_17, %none_18 : !torch.vtensor<[2,4],si32>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[2,4],f32>
    %int1_19 = torch.constant.int 1
    %13 = torch.aten.unsqueeze %4, %int1_19 : !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[2,1],f32>
    %14 = torch.aten.mul.Tensor %12, %13 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,1],f32> -> !torch.vtensor<[2,4],f32>
    %15 = torch.vtensor.literal(dense_resource<torch_tensor_4_torch.float32> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %int0_20 = torch.constant.int 0
    %16 = torch.aten.unsqueeze %15, %int0_20 : !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[1,4],f32>
    %17 = torch.aten.mul.Tensor %14, %16 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[1,4],f32> -> !torch.vtensor<[2,4],f32>
    %18 = torch.vtensor.literal(dense_resource<torch_tensor_4_torch.float32_1> : tensor<4xf32>) : !torch.vtensor<[4],f32>
    %int1_21 = torch.constant.int 1
    %19 = torch.aten.add.Tensor %17, %18, %int1_21 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %20 = torch.aten.relu %19 : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    return %20 : !torch.vtensor<[2,4],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_8_4_torch.int8: "0x01000000FFF43CEA53235C0E81D8987C8EE6BD81C58138AA29A87FDDFDC9E1CB7A057276",
      torch_tensor_4_torch.float32: "0x040000000329163B5B4C2E3BF77F173B9F3F293B",
      torch_tensor_4_torch.float32_1: "0x0400000086AA6ABE86A826BE53EF7CBE2089A9BE"
    }
  }
#-}
