#include <TFile.h>
#include <TTree.h>
#include <TLeaf.h>
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
#include <cmath>
#include <limits>

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
                     const char *source_path = "", const char *raw_scores_path = "") {
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
        const bool has_raw = std::string(raw_scores_path).size() != 0;
        if (has_raw && source->GetBranch("Score_raw"))
            throw std::runtime_error("Score_raw branch collision");
        if (auto *legacy = source->GetBranch("Score")) {
            if (source->GetBranch("Score_source"))
                throw std::runtime_error("Score and Score_source both exist");
            auto *leaf = legacy->GetLeaf("Score");
            const std::string title = legacy->GetTitle();
            if (!leaf || legacy->GetListOfLeaves()->GetEntries() != 1 ||
                title.rfind("Score/", 0) != 0)
                throw std::runtime_error("Unsupported legacy Score branch layout");
            // Preserve the input score; reserve Score for this model's output.
            legacy->SetName("Score_source");
            legacy->SetTitle(("Score_source" + title.substr(5)).c_str());
            leaf->SetName("Score_source");
            leaf->SetTitle("Score_source");
        }
        std::ifstream input(scores_path, std::ios::binary);
        std::ifstream raw_input;
        if (has_raw) raw_input.open(raw_scores_path, std::ios::binary);
        float raw = std::numeric_limits<float>::quiet_NaN();
        float score = -1.f;
        file.cd();
        TTree *target = filter_data ? source->CloneTree(0) : source;
        auto *branch = target->Branch("Score", &score, "Score/F");
        auto *raw_branch = has_raw ? target->Branch("Score_raw", &raw, "Score_raw/F") : nullptr;
        const auto entries = source->GetEntries();
        for (Long64_t i = 0; i < entries; ++i) {
            if (!input.read(reinterpret_cast<char *>(&score), sizeof(score)))
                throw std::runtime_error("Cannot read score stream");
            if (has_raw) {
                if (!raw_input.read(reinterpret_cast<char*>(&raw), sizeof(raw)))
                    throw std::runtime_error("Cannot read raw score stream");
                if (!std::isfinite(score) || (score != -1.f && (score < 0.f || score > 1.f)))
                    throw std::runtime_error("Invalid mapped score");
                if (score >= 0.f && (!std::isfinite(raw) || raw < -1.f || raw > 1.f ||
                    std::abs(score - (raw+1.f)/2.f) > 2.e-7f))
                    throw std::runtime_error("Invalid raw/mapped score pair");
                if (score == -1.f && !std::isnan(raw))
                    throw std::runtime_error("Unscored raw response must be NaN");
            }
            if (filter_data) {
                if (score < 0.f) continue;
                source->GetEntry(i);
                target->Fill();
            } else {
                branch->Fill();
                if (raw_branch) raw_branch->Fill();
            }
        }
        char extra;
        if (input.read(&extra, 1) || (has_raw && raw_input.read(&extra, 1)))
            throw std::runtime_error("Extra score stream entries");
        target->Write(tree_name, TObject::kOverwrite);
        std::cout << "Saved " << tree_name << ": " << target->GetEntries()
                  << " RECO entries" << std::endl;
        file.Close();
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        gSystem->Exit(1);
    }
}
