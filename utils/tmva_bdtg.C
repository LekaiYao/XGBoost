#include <TMVA/MethodBDT.h>
#include <TMVA/DecisionTree.h>
#include <algorithm>
#include <TMVA/Tools.h>
#include <TMVA/Factory.h>
#include <TMVA/DataLoader.h>
#include <TMVA/DataSet.h>
#include <TMVA/Event.h>
#include <TMVA/Reader.h>
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>
#include <TSystem.h>
#include <TCut.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
R__LOAD_LIBRARY(libTMVA)

void tmva_bdtg(const char *mode, const char *input, const char *output,
               const char *columns, const char *options="", const char *xml="",
               int seed=42, double fraction=.75) {
 try {
    std::vector<std::string> names; std::stringstream ss(columns); std::string n;
    while (std::getline(ss,n,',')) names.push_back(n);
    if (names.empty()) throw std::runtime_error("No TMVA variables");
    TMVA::Tools::Instance(); gRandom->SetSeed(seed);
    if (std::string(mode)=="predict") {
        TMVA::Reader reader("!Color:!Silent"); std::vector<float> row(names.size());
        for (size_t i=0;i<names.size();++i) reader.AddVariable(names[i],&row[i]);
        float row_index=0; reader.AddSpectator("row_index", &row_index);
        reader.BookMVA("BDTG",xml);
        std::ifstream in(input,std::ios::binary); std::ofstream out(output,std::ios::binary);
        if (!in || !out) throw std::runtime_error("Cannot open TMVA binary streams");
        while (in.read(reinterpret_cast<char*>(row.data()),row.size()*sizeof(float))) {
            double raw=reader.EvaluateMVA("BDTG");
            if (!std::isfinite(raw) || raw < -1. || raw > 1.) throw std::runtime_error("Invalid BDTG response");
            out.write(reinterpret_cast<char*>(&raw),sizeof(raw));
        }
        if (in.gcount()!=0 || !out) throw std::runtime_error("Incomplete TMVA binary stream");
        return;
    }
    if (std::string(mode)=="diagnostics") {
        TFile source(input,"READ"),result(output,"RECREATE");
        auto *tree=source.Get<TTree>("events");
        if (!tree || source.IsZombie() || result.IsZombie()) throw std::runtime_error("Invalid diagnostics ROOT");
        std::vector<float> row(names.size());
        TMVA::Reader reader("!Color:!Silent");
        for(size_t i=0;i<names.size();++i) {tree->SetBranchAddress(names[i].c_str(),&row[i]);reader.AddVariable(names[i],&row[i]);}
        float spectator=0;reader.AddSpectator("row_index",&spectator);
        auto *method=dynamic_cast<TMVA::MethodBDT*>(reader.BookMVA("BDTG",xml));
        if(!method) throw std::runtime_error("BDTG reader type mismatch");
        bool label=false,is_train=false;double weight=0;
        tree->SetBranchAddress("label",&label);tree->SetBranchAddress("is_train",&is_train);tree->SetBranchAddress("physical_weight",&weight);
        std::vector<TMVA::Event> events;std::vector<double> weights,margin;std::vector<bool> labels,trains;
        const auto size=tree->GetEntries();events.reserve(size);double sums[2]={0,0};
        for(Long64_t i=0;i<size;++i) {
            tree->GetEntry(i);
            if(!std::isfinite(weight)||weight<=0)throw std::runtime_error("Invalid diagnostic weight");
            events.emplace_back(row,0);weights.push_back(weight);labels.push_back(label);trains.push_back(is_train);sums[is_train?0:1]+=weight;
        }
        if(sums[0]<=0||sums[1]<=0)throw std::runtime_error("Empty diagnostic split");
        margin.assign(size,0.);result.cd();
        TTree history("history","Physical-weight logloss at every saved boosting round");
        int round=0;double train_loss=0,test_loss=0;
        history.Branch("round",&round);history.Branch("train_logloss",&train_loss);history.Branch("test_logloss",&test_loss);
        for(auto *dt:method->GetForest()) {
            double loss[2]={0,0};
            for(Long64_t i=0;i<size;++i) {
                margin[i]+=2*dt->CheckEvent(&events[i],false);
                const auto m=margin[i];
                loss[trains[i]?0:1]+=weights[i]*(std::max(m,0.)-(labels[i]?m:0.)+std::log1p(std::exp(-std::abs(m))));
            }
            ++round;train_loss=loss[0]/sums[0];test_loss=loss[1]/sums[1];history.Fill();
        }
        history.Write();
        TTree scores("scores","Final additive margin and native raw response in source row order");double m=0,raw=0;
        scores.Branch("margin",&m);scores.Branch("raw",&raw);
        for(auto value:margin){m=value;raw=std::tanh(m/2);scores.Fill();}scores.Write();return;
    }
    if (std::string(mode)!="train") throw std::runtime_error("Unknown TMVA operation");
    TFile source(input,"READ"); TFile result(output,"RECREATE");
    if(source.IsZombie() || result.IsZombie()) throw std::runtime_error("Cannot open training ROOT files");
    auto *sig=source.Get<TTree>("signal");auto *bkg=source.Get<TTree>("background");
    if (!sig || !bkg) throw std::runtime_error("Missing training trees");
    TMVA::DataLoader loader("dataset");
    TMVA::Factory factory("ML",&result,"!V:!Silent:AnalysisType=Classification:Transformations=I");
    for (const auto &name:names) loader.AddVariable(name,'F');
    loader.AddSpectator("row_index",'F');
    loader.AddSignalTree(sig,1.); loader.AddBackgroundTree(bkg,1.);
    loader.SetSignalWeightExpression("training_weight");loader.SetBackgroundWeightExpression("training_weight");
    auto ns=sig->GetEntries(),nb=bkg->GetEntries();
    auto nts=static_cast<Long64_t>(fraction*ns),ntb=static_cast<Long64_t>(fraction*nb);
    if (nts<2 || ntb<2 || nts>=ns || ntb>=nb) throw std::runtime_error("Too few events for TMVA split");
    TString split=Form("nTrain_Signal=%lld:nTest_Signal=%lld:nTrain_Background=%lld:nTest_Background=%lld:SplitMode=Random:SplitSeed=%d:NormMode=None:!V",nts,ns-nts,ntb,nb-ntb,seed);
    loader.PrepareTrainingAndTestTree(TCut(""),TCut(""),split);
    // Keep the trained object alive until native importance is exported.
    // Persist the method normally; Factory's automatic delete/reload is disabled.
    auto *method=dynamic_cast<TMVA::MethodBDT*>(factory.BookMethod(&loader,TMVA::Types::kBDT,"BDTG",options));
    if(!method)throw std::runtime_error("Expected BDTG method");
    factory.SetOptions("ModelPersistence=False");factory.ParseOptions();
    factory.TrainAllMethods();
    result.cd();TTree importance("native_importance","TMVA GetVariableImportance from live trained forest");
    int feature=0;double value=0;importance.Branch("feature_index",&feature);importance.Branch("importance",&value);
    const auto values=method->GetVariableImportance();
    for(size_t i=0;i<values.size();++i){feature=i;value=values[i];if(!std::isfinite(value)||value<0)throw std::runtime_error("Invalid native importance");importance.Fill();}
    importance.Write();
    factory.TestAllMethods();factory.EvaluateAllMethods();
    result.cd();
    TTree identities("split","Actual TMVA members and training weights");
    Long64_t index=0;bool train=false;double weight=0;
    identities.Branch("row_index",&index);identities.Branch("is_train",&train);identities.Branch("training_weight",&weight);
    for (auto type:{TMVA::Types::kTraining,TMVA::Types::kTesting}) {
        train=type==TMVA::Types::kTraining;
        for (const auto *event:loader.GetDataSetInfo().GetDataSet()->GetEventCollection(type)) {
            index=std::llround(event->GetSpectator(0));weight=event->GetOriginalWeight();identities.Fill();
        }
    }
    identities.Write();
 } catch (const std::exception &e) {std::cerr<<e.what()<<std::endl;gSystem->Exit(1);}
}
