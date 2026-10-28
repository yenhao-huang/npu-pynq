#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir;

static std::string typeText(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

static llvm::json::Value attributeValue(Attribute attribute) {
  if (auto value = dyn_cast<StringAttr>(attribute))
    return value.getValue().str();
  if (auto value = dyn_cast<IntegerAttr>(attribute))
    return value.getValue().getSExtValue();
  if (auto value = dyn_cast<FloatAttr>(attribute))
    return value.getValueAsDouble();
  if (auto value = dyn_cast<TypeAttr>(attribute))
    return typeText(value.getValue());
  if (auto value = dyn_cast<ArrayAttr>(attribute)) {
    llvm::json::Array elements;
    for (Attribute element : value)
      elements.push_back(attributeValue(element));
    return elements;
  }
  if (auto value = dyn_cast<DenseElementsAttr>(attribute)) {
    llvm::json::Object descriptor;
    descriptor["type"] = typeText(value.getType());
    descriptor["elements"] = value.getNumElements();
    if (value.getNumElements() <= 16) {
      llvm::json::Array elements;
      if (isa<IntegerType>(value.getElementType())) {
        for (const APInt &element : value.getValues<APInt>())
          elements.push_back(element.getSExtValue());
      } else if (isa<FloatType>(value.getElementType())) {
        for (const APFloat &element : value.getValues<APFloat>())
          elements.push_back(element.convertToDouble());
      }
      descriptor["values"] = std::move(elements);
    }
    return descriptor;
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << "Usage: onnx-import-adapter INPUT.mlir OUTPUT.json\n";
    return 2;
  }
  DialectRegistry registry;
  registerAllDialects(registry);
  MLIRContext context(registry);
  context.allowUnregisteredDialects();
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module)
    return 1;
  llvm::json::Object graph;
  graph["format"] = "onnx-import-graph/1";
  graph["mlir_major"] = LLVM_VERSION_MAJOR;
  if (auto version = module->getOperation()->getAttrOfType<StringAttr>("onnx-mlir.compiler_version"))
    graph["importer_version"] = version.getValue().str();
  llvm::DenseMap<Value, int64_t> values;
  int64_t nextValue = 0;
  llvm::json::Array functions;
  for (auto function : module->getOps<func::FuncOp>()) {
    if (function.isExternal())
      continue;
    llvm::json::Object description;
    description["name"] = function.getSymName().str();
    llvm::json::Array arguments;
    for (BlockArgument argument : function.getArguments()) {
      int64_t identifier = nextValue++;
      values[argument] = identifier;
      llvm::json::Object input;
      input["id"] = identifier;
      input["type"] = typeText(argument.getType());
      if (auto name = function.getArgAttrOfType<StringAttr>(argument.getArgNumber(), "onnx.name"))
        input["name"] = name.getValue().str();
      arguments.push_back(std::move(input));
    }
    description["inputs"] = std::move(arguments);
    llvm::json::Array nodes;
    function.walk([&](Operation *operation) {
      if (operation == function.getOperation())
        return;
      llvm::json::Object node;
      node["op"] = operation->getName().getStringRef().str();
      llvm::json::Array operands;
      for (Value operand : operation->getOperands()) {
        auto found = values.find(operand);
        if (found == values.end()) {
          llvm::errs() << "Unsupported non-topological or nested graph\n";
          std::exit(1);
        }
        operands.push_back(found->second);
      }
      llvm::json::Array results;
      for (Value result : operation->getResults()) {
        int64_t identifier = nextValue++;
        values[result] = identifier;
        results.push_back(llvm::json::Object{{"id", identifier}, {"type", typeText(result.getType())}});
      }
      llvm::json::Object attributes;
      for (NamedAttribute attribute : operation->getAttrs())
        attributes[attribute.getName().str()] = attributeValue(attribute.getValue());
      if (auto properties = dyn_cast_if_present<DictionaryAttr>(operation->getPropertiesAsAttribute()))
        for (NamedAttribute property : properties)
          attributes[property.getName().str()] = attributeValue(property.getValue());
      node["inputs"] = std::move(operands);
      node["outputs"] = std::move(results);
      node["attributes"] = std::move(attributes);
      nodes.push_back(std::move(node));
    });
    description["nodes"] = std::move(nodes);
    functions.push_back(std::move(description));
  }
  graph["functions"] = std::move(functions);
  std::error_code error;
  llvm::raw_fd_ostream output(argv[2], error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << error.message() << "\n";
    return 1;
  }
  output << llvm::formatv("{0:2}", llvm::json::Value(std::move(graph))) << "\n";
  return 0;
}
