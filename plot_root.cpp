#include <TFile.h>
#include <TTree.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <iostream>
#include <cmath>

// Function to plot variance maps
void plot_variance_maps(const char* filename) {
    // Open the ROOT file
    TFile *file = TFile::Open(filename);
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Access the tree inside the file
    TTree *tree = (TTree*)file->Get("tree");
    if (!tree) {
        std::cerr << "Error: Cannot find TTree named 'tree' in " << filename << std::endl;
        return;
    }

    // Declare variables for the branches
    float x, y, z;
    double angle;
    tree->SetBranchAddress("x", &x);
    tree->SetBranchAddress("y", &y);
    tree->SetBranchAddress("z", &z);
    tree->SetBranchAddress("angleDev", &angle);

    const int nbins = 100;
    float xmin = -600, xmax = 600;
    float ymin = -600, ymax = 600;
    float zmin = -600, zmax = 600;

    // Create C-style arrays to hold sum, sum of squares, and counts for each bin
    float sum_xy[nbins][nbins] = {{0}};
    float sumsq_xy[nbins][nbins] = {{0}};
    int count_xy[nbins][nbins] = {{0}};

    float sum_yz[nbins][nbins] = {{0}};
    float sumsq_yz[nbins][nbins] = {{0}};
    int count_yz[nbins][nbins] = {{0}};

    float sum_xz[nbins][nbins] = {{0}};
    float sumsq_xz[nbins][nbins] = {{0}};
    int count_xz[nbins][nbins] = {{0}};

    // Loop over the entries in the tree
    Long64_t nentries = tree->GetEntries();
    for (Long64_t i = 0; i < nentries; ++i) {
        tree->GetEntry(i);

        int binx = (int)((x - xmin) / (xmax - xmin) * nbins);
        int biny = (int)((y - ymin) / (ymax - ymin) * nbins);
        int binz = (int)((z - zmin) / (zmax - zmin) * nbins);

        if (binx >= 0 && binx < nbins && biny >= 0 && biny < nbins) {
            sum_xy[binx][biny] += angle;
            sumsq_xy[binx][biny] += angle * angle;
            count_xy[binx][biny]++;
        }
        if (biny >= 0 && biny < nbins && binz >= 0 && binz < nbins) {
            sum_yz[biny][binz] += angle;
            sumsq_yz[biny][binz] += angle * angle;
            count_yz[biny][binz]++;
        }
        if (binx >= 0 && binx < nbins && binz >= 0 && binz < nbins) {
            sum_xz[binx][binz] += angle;
            sumsq_xz[binx][binz] += angle * angle;
            count_xz[binx][binz]++;
        }
    }

    // Create histograms for variance
    TH2F *hXY = new TH2F("hXY", "Variance of angle in x-y", nbins, xmin, xmax, nbins, ymin, ymax);
    TH2F *hYZ = new TH2F("hYZ", "Variance of angle in y-z", nbins, ymin, ymax, nbins, zmin, zmax);
    TH2F *hXZ = new TH2F("hXZ", "Variance of angle in x-z", nbins, xmin, xmax, nbins, zmin, zmax);

    // Fill histograms with variance data
    for (int i = 0; i < nbins; ++i) {
        for (int j = 0; j < nbins; ++j) {
            if (count_xy[i][j] > 1) {
                float mean = sum_xy[i][j] / count_xy[i][j];
                float mean2 = sumsq_xy[i][j] / count_xy[i][j];
                float variance  = mean2 - mean * mean;
                float sd = (variance > 0) ? sqrtf(variance) : 0;
                hXY->SetBinContent(i+1, j+1, sd);
            }
            if (count_yz[i][j] > 1) {
                float mean = sum_yz[i][j] / count_yz[i][j];
                float mean2 = sumsq_yz[i][j] / count_yz[i][j];
                float variance  = mean2 - mean * mean;
                float sd = (variance > 0) ? sqrtf(variance) : 0;
                hYZ->SetBinContent(i+1, j+1, sd);
            }
            if (count_xz[i][j] > 1) {
                float mean = sum_xz[i][j] / count_xz[i][j];
                float mean2 = sumsq_xz[i][j] / count_xz[i][j];
                float variance  = mean2 - mean * mean;
                float sd = (variance > 5e-4) ? sqrtf(variance) : 0;
                hXZ->SetBinContent(i+1, j+1, sd);
            }
        }
    }

    // Create canvas and draw histograms
    TCanvas *c1 = new TCanvas("c1", "Angle Variance Maps", 1920, 1080);
    gStyle->SetPalette(kVisibleSpectrum); // Rainbow color palette
    //c1->Divide(3,1);
    //c1->cd(1); hXY->Draw("COLZ");
    hXZ->SetContour(20);  // 20 color levels
    hXZ->SetStats(0);
    //c1->cd(2); hYZ->Draw("COLZ");
    c1->cd(3); hXZ->Draw("COLZ");
}
