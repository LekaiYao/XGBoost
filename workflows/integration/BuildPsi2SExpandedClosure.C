#include <TBranch.h>
#include <TFile.h>
#include <TLeaf.h>
#include <TSystem.h>
#include <TTree.h>

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const std::vector<std::string> kCurrent = {
    "Bmass", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb",
    "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt",
    "Bpt", "By", "BQvalue", "Reweight", "Prediction"};
const std::vector<std::string> kIdentity = {
    "Bmass", "Bchi2Prob", "Btrk1dR", "Btrk2dR", "BtrkPtimb",
    "Btrk1Pt", "Btrk2Pt", "BtktkvProb", "Bcos_dtheta", "Btktkpt",
    "Bpt", "By", "BQvalue"};
const std::vector<std::string> kAdded = {
    "BujvProb", "Balpha", "Btrk2Eta", "Btrk1Eta", "Btrk1Phi",
    "Btrk2Phi", "Bmu1y", "Bmu2y", "Bmu1pt", "Bmu2pt"};

struct Scalar {
  std::string type;
  float f = 0;
  double d = 0;

  void *address() {
    if (type == "Float_t") return &f;
    if (type == "Double_t") return &d;
    throw std::runtime_error("Unsupported scalar type: " + type);
  }
  const void *data() const { return type == "Float_t" ? static_cast<const void *>(&f) : static_cast<const void *>(&d); }
  std::size_t size() const { return type == "Float_t" ? sizeof(f) : sizeof(d); }
};

TTree *RequireTree(TFile &file, const char *name) {
  auto *tree = dynamic_cast<TTree *>(file.Get(name));
  if (!tree) throw std::runtime_error(std::string("Missing TTree: ") + name);
  return tree;
}

Scalar Bind(TTree *tree, const std::string &name) {
  auto *branch = tree->GetBranch(name.c_str());
  if (!branch) throw std::runtime_error("Missing branch: " + name);
  auto *leaf = branch->GetLeaf(name.c_str());
  if (!leaf || leaf->GetLenStatic() != 1) throw std::runtime_error("Branch is not a scalar: " + name);
  Scalar value;
  value.type = leaf->GetTypeName();
  tree->SetBranchAddress(name.c_str(), value.address());
  return value;
}

bool Same(const Scalar &left, const Scalar &right) {
  return left.type == right.type && left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size()) == 0;
}

std::vector<Scalar> BindMany(TTree *tree, const std::vector<std::string> &names) {
  std::vector<Scalar> values;
  values.reserve(names.size());
  for (const auto &name : names) values.push_back(Bind(tree, name));
  // Bind again after vector storage is stable: SetBranchAddress must point to
  // the final Scalar objects rather than temporaries returned by Bind.
  for (std::size_t i = 0; i < names.size(); ++i)
    tree->SetBranchAddress(names[i].c_str(), values[i].address());
  return values;
}
}  // namespace

void BuildPsi2SExpandedClosure(const char *rawPath, const char *currentPath,
                               const char *outputPath,
                               const char *treeName = "ntmix_PSI2S") {
  try {
    TFile rawFile(rawPath, "READ");
    TFile currentFile(currentPath, "READ");
    if (rawFile.IsZombie() || currentFile.IsZombie())
      throw std::runtime_error("Unable to open an input ROOT file");
    auto *raw = RequireTree(rawFile, treeName);
    auto *current = RequireTree(currentFile, treeName);
    if (raw->GetEntries() != current->GetEntries())
      throw std::runtime_error("Raw/current entry-count mismatch");

    auto rawIdentity = BindMany(raw, kIdentity);
    auto currentIdentity = BindMany(current, kIdentity);
    auto rawAdded = BindMany(raw, kAdded);
    for (std::size_t i = 0; i < kAdded.size(); ++i)
      if (rawAdded[i].type != "Float_t")
        throw std::runtime_error("Added branch is not Float_t: " + kAdded[i]);

    TFile outputFile(outputPath, "RECREATE");
    if (outputFile.IsZombie()) throw std::runtime_error("Unable to create output ROOT file");
    outputFile.cd();
    auto *output = current->CloneTree(0);
    output->SetName(treeName);
    for (std::size_t i = 0; i < kAdded.size(); ++i)
      output->Branch(kAdded[i].c_str(), &rawAdded[i].f, (kAdded[i] + "/F").c_str());

    const auto entries = current->GetEntries();
    for (Long64_t entry = 0; entry < entries; ++entry) {
      current->GetEntry(entry);
      raw->GetEntry(entry);
      for (std::size_t i = 0; i < kIdentity.size(); ++i) {
        if (!Same(rawIdentity[i], currentIdentity[i]))
          throw std::runtime_error("Entry-order identity mismatch at entry " +
                                   std::to_string(entry) + " branch " + kIdentity[i]);
      }
      output->Fill();
    }
    output->Write();
    outputFile.Close();

    TFile checkFile(outputPath, "READ");
    auto *check = RequireTree(checkFile, treeName);
    if (check->GetEntries() != entries || check->GetListOfBranches()->GetEntries() != 25)
      throw std::runtime_error("Expanded output entries/branch count mismatch");
    auto checkCurrent = BindMany(check, kCurrent);
    auto sourceCurrent = BindMany(current, kCurrent);
    auto checkAdded = BindMany(check, kAdded);
    auto sourceAdded = BindMany(raw, kAdded);
    for (Long64_t entry = 0; entry < entries; ++entry) {
      check->GetEntry(entry);
      current->GetEntry(entry);
      raw->GetEntry(entry);
      for (std::size_t i = 0; i < kCurrent.size(); ++i)
        if (!Same(checkCurrent[i], sourceCurrent[i]))
          throw std::runtime_error("Output changed current branch " + kCurrent[i]);
      for (std::size_t i = 0; i < kAdded.size(); ++i)
        if (!Same(checkAdded[i], sourceAdded[i]))
          throw std::runtime_error("Output changed added branch " + kAdded[i]);
    }
    std::cout << "EXPANDED_CLOSURE_OK entries=" << entries << " branches=25" << std::endl;
  } catch (const std::exception &error) {
    std::cerr << "EXPANDED_CLOSURE_ERROR " << error.what() << std::endl;
    gSystem->Exit(2);
  }
}
