#include <TFile.h>
#include <TTree.h>
#include <TObject.h>
#include <TSystem.h>
#include <TKey.h>
#include <TDirectory.h>
#include <memory>
#include <set>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>

// Copy live objects into a fresh file, without obsolete/free ROOT data blocks.
void full_apply_copy_objects(TDirectory *source, TDirectory *target,
                             const std::string &skip = "") {
    std::set<std::string> copied;
    TIter next(source->GetListOfKeys());
    while (auto *key = dynamic_cast<TKey *>(next())) {
        const std::string name = key->GetName();
        if (name == skip || !copied.insert(name).second) continue;
        auto *object = source->Get(name.c_str());
        target->cd();
        if (auto *directory = dynamic_cast<TDirectory *>(object)) {
            auto *child = target->mkdir(name.c_str(), directory->GetTitle());
            full_apply_copy_objects(directory, child);
        } else if (auto *tree = dynamic_cast<TTree *>(object)) {
            auto *copy = tree->CloneTree(-1, "fast");
            copy->Write(name.c_str(), TObject::kOverwrite);
        } else {
            object->Write(name.c_str(), TObject::kOverwrite);
        }
    }
    target->cd();
}

// MC: update a complete copy. DATA: build a fresh file from the source.
void full_apply_root(const char *path, const char *tree_name,
                     const char *scores_path, bool filter_data,
                     const char *source_path = "") {
    try {
        std::unique_ptr<TFile> original;
        if (filter_data) {
            original.reset(TFile::Open(source_path, "READ"));
            if (!original || original->IsZombie())
                throw std::runtime_error("Cannot open DATA source");
        }
        TFile file(path, filter_data ? "RECREATE" : "UPDATE");
        if (file.IsZombie()) throw std::runtime_error("Cannot open ROOT output");
        if (filter_data) {
            file.SetCompressionSettings(original->GetCompressionSettings());
            full_apply_copy_objects(original.get(), &file, tree_name);
        }
        auto *source = dynamic_cast<TTree *>(
            (filter_data ? original.get() : &file)->Get(tree_name));
        if (!source) throw std::runtime_error("Missing RECO TTree");
        if (source->GetBranch("Score")) throw std::runtime_error("Score already exists");
        std::ifstream input(scores_path, std::ios::binary);
        float score = -1.f;
        file.cd();
        TTree *target = filter_data ? source->CloneTree(0) : source;
        auto *branch = target->Branch("Score", &score, "Score/F");
        const auto entries = source->GetEntries();
        for (Long64_t i = 0; i < entries; ++i) {
            if (!input.read(reinterpret_cast<char *>(&score), sizeof(score)))
                throw std::runtime_error("Cannot read score stream");
            if (filter_data) {
                if (score < 0.f) continue;
                source->GetEntry(i);
                target->Fill();
            } else {
                branch->Fill();
            }
        }
        target->Write(tree_name, TObject::kOverwrite);
        std::cout << "Saved " << tree_name << ": " << target->GetEntries()
                  << " RECO entries" << std::endl;
        file.Close();
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        gSystem->Exit(1);
    }
}
