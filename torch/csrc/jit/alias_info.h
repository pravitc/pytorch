#pragma once
#include <unordered_set>
#include <vector>
#include "torch/csrc/jit/interned_strings.h"

namespace torch {
namespace jit {

class AliasInfo {
 public:
  // Symbol for the set that can alias
  static Symbol wildcard() {
    static const Symbol wc = Symbol::fromQualString("alias::*");
    return wc;
  }

  AliasInfo() {}

  void setIsWrite(bool isWrite) {
    isWrite_ = isWrite;
  }
  bool isWrite() const {
    return isWrite_;
  }

  void addInputSet(Symbol aliasSet) {
    inputSets_.insert(aliasSet);
  }
  void addOutputSet(Symbol aliasSet) {
    outputSets_.insert(aliasSet);
  }
  // At the beginning of this op, which alias sets does this value belong to?
  const std::unordered_set<Symbol>& inputSets() const {
    return inputSets_;
  }
  // At the end of this op, which alias sets does this value belong to?
  // This can change if, e.g. we are appending to a list:
  //   aten::append(Tensor(b -> b|c)[](a!) list, Tensor(c) el)
  const std::unordered_set<Symbol>& outputSets() const {
    // In the common case, the alias sets that this value belong to don't change
    // during the op, so just return the input set
    if (outputSets_.empty()) {
      return inputSets_;
    }
    return outputSets_;
  }
  // the alias info for the contained types of the type
  // e.g. if this is an annotation on List[T], `sets` refers to
  // the alias sets that the list may be in
  // while containedTypes()[0] refers to the sets that members of the list
  // may be in
  void addContainedType(AliasInfo aliasInfo) {
    containedTypes_.push_back(std::move(aliasInfo));
  }
  const std::vector<AliasInfo>& containedTypes() const {
    return containedTypes_;
  }

 private:
  std::unordered_set<Symbol> inputSets_;
  std::unordered_set<Symbol> outputSets_;
  std::vector<AliasInfo> containedTypes_;
  bool isWrite_ = false;
};

inline std::ostream& operator<<(std::ostream& out, const AliasInfo& info) {
  out << "(";

  bool first = true;
  for (const auto& aliasSet : info.inputSets()) {
    if (first) {
      first = false;
    } else {
      out << "|";
    }
    out << aliasSet.toUnqualString();
  }

  if (!info.containedTypes().empty()) {
    out << "[";
    for (size_t i = 0; i < info.containedTypes().size(); i++) {
      if (i > 0) {
        out << ", ";
      }
      out << info.containedTypes()[i];
    }
    out << "]";
  }

  out << ")";
  return out;
}

} // namespace jit
} // namespace torch
