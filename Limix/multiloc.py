#!/bin/python

'''
Script to run GWAS with specified locus as fixed effect
specifically for collaboration with BOKU

requires 10GB  memory, maybe less
srun --mem 10GB --pty bash
'''


from limix.qtl import scan
import pandas as pd
import numpy as np
import h5py
from bisect import bisect
from limix import plot
import random
from sklearn.decomposition import PCA
import argparse
import os
from statsmodels.stats import multitest
import math

# Parameters
parser = argparse.ArgumentParser(description = 'Parse parameters for multilocus GWAS')
parser.add_argument('-p', '--phenotype', help = 'Path to phenotype file. CSV file with accession ID in forst column, phenptype values in second column. Filename is used as phenotype name.', required = True)
parser.add_argument('-g', '--genotype', help = 'Path to genotype directory.This directory should contain boht the SNP and kinship matrix. Versions are the same as used for PyGWAS.', required = True)
parser.add_argument('-l', '--locus', help = 'Specify the locus to use as fixed factor. Takes two argument: first is the chromosome number, second is the position on that chromosome', nargs = 2, required = True)
parser.add_argument('-m', '--maf', help = 'Specify the minor allele frequecny cut-off. Default is set to 0.05', default = 0.05)
parser.add_argument('-o', '--outDir', help = 'Specify the output directory. All results will be saved in this directory.', required = True)
args = parser.parse_args()






# Phenotype (Y)
pheno = pd.read_csv(args.phenotype, index_col = 0)
trait = os.path.basename(args.phenotype)[:-4] 
# remove NA values
pheno = pheno[np.isfinite(pheno)]
# enocde pheno.index to UTF8, for complemetarity with SNP matrix accessions
pheno.index = pheno.index.map(lambda x: str(x).encode('UTF8'))
acnNrInitial = len(pheno.index)

# Genotype (G)
genoFile = f'{args.genotype}/all_chromosomes_binary.hdf5'

geno_hdf = h5py.File(genoFile, 'r')

acn_indices = [np.where(geno_hdf['accessions'][:] == acn)[0][0] for acn in pheno.index]
acn_indices.sort()
acn_order = geno_hdf['accessions'][acn_indices]
G = geno_hdf['snps'][:, acn_indices]
# subset pheno in case of non-genotyped accessions
if len(acn_indices) != len(pheno.index):
    acns = list(set(pheno.index) & set(geno_hdf['accessions'][:]))
    pheno = pheno[acns]
# select only SNPs with minor allele frequecny above threshold
AC1 = G.sum(axis = 1)
AC0 = G.shape[1] - AC1
AC = np.vstack((AC0, AC1))
MAC = np.min(AC, axis = 0)
MAF = MAC/G.shape[1]
SNP_indices =  np.where(MAF >= float(args.maf))[0]
SNPs_MAF = MAF[SNP_indices] 
G = G[SNP_indices, :]
print(f'Of the {acnNrInitial} phenotyped accessions, {len(acn_indices)} accessions were present in the SNP matrix.')
print(f'{len(SNP_indices)} SNPs had a minor allele frequency higher than {args.maf}')
# transpose G matrix into the required acccessions x SNPs format
G = G.transpose()

# Covariates (M)
covarLocus = tuple(map(int, args.locus))
chrIdx = geno_hdf['positions'].attrs['chr_regions']
# get indices for covariate chromosome
covarChr_Idx = chrIdx[covarLocus[0] - 1]
# get the index in the SNP matrix of the covariate locus
covarLocus_Idx = np.where(geno_hdf['positions'][covarChr_Idx[0]:covarChr_Idx[1]] == covarLocus[1])[0][0] + covarChr_Idx[0]
M = geno_hdf['snps'][covarLocus_Idx, :][acn_indices]
# get MAF and check if it is higher than the set threshold
AC1 = M.sum()
AC = [AC1, len(M) - AC1]
MAF = min(AC)/len(M)
if MAF < float(args.maf):
    print(f'CAREFUL: the selected covariate locus has a minor allele frequency of {MAF}, which is below the set threshold of {args.maf}')
geno_hdf.close() 

# Kinship (K)
kinFile = f'{args.genotype}/kinship_ibs_binary_mac5.h5py'
kin_hdf = h5py.File(kinFile, 'r')
# select kinship only for phenotyped and genotyped accessions
acn_indices = [np.where(kin_hdf['accessions'][:] == acn)[0][0] for acn in pheno.index]
acn_indices.sort()
K = kin_hdf['kinship'][acn_indices, :][:, acn_indices]
kin_hdf.close()

# get phenotype in correct order
pheno = pheno.loc[acn_order]
Y = pheno.to_numpy()


# scan
r = scan(G, Y, K = K, lik = "normal", M=M, verbose = True)

# save results
# link chromosome and positions to p-values and effect sizes
chrom = [bisect(chrIdx[:, 1], snpIdx) + 1 for snpIdx in SNP_indices]
geno_hdf = h5py.File(genoFile, 'r')
positions = geno_hdf['positions'][:]
pos = [positions[snp] for snp in SNP_indices]
pvalues = r.stats.pv20.tolist()

Bonferroni = multitest.multipletests(pvalues, alpha = 0.05, method = 'fdr_bh')[3]

gwas_tuples = list(zip(chrom, pos, pvalues))
gwas_results = pd.DataFrame(gwas_tuples, columns = ['chrom', 'pos', 'pv'])

# plot results
# Manhattan plot
plot.manhattan(gwas_results)
plt = plot.get_pyplot() 
_ = plt.axhline(-math.log(Bonferroni, 10), color='red')
plt.savefig(f'{args.outDir}/manhattanPlot_{trait}_multiLocus_chr{covarLocus[0]}_pos{covarLocus[1]}_{args.maf}.png')
plt.close()

# QQ-plot
plot.qqplot(gwas_results.pv)
plt = plot.get_pyplot()
plt.savefig(f'{args.outDir}/qqPlot_{trait}_multiLocus_chr{covarLocus[0]}_pos{covarLocus[1]}_{args.maf}.png')
plt.close()

# Save results, results cna be uploaded in GWAS portal
#TODO: chr pos pvalue maf mac GVE
gwas_results['maf'] = SNPs_MAF
gwas_results['mac'] = gwas_results.maf * len(acn_indices)
gwas_results.mac = gwas_results.mac.astype(int)
gwas_results['GVE'] = np.nan
gwas_results.columns.values[0] = 'chr' 
gwas_results.columns.values[2] = 'pvalue'

gwas_results.to_csv(f'{args.outDir}/{trait}_multilocus_chr{covarLocus[0]}_pos{covarLocus[1]}_{args.maf}.csv', index = False)



'''

# single trait
r_single = scan(G = G, Y = Y, K = K, lik = 'normal', M = None, verbose = True)

chrom = [bisect(chrIdx[:, 1], snpIdx) + 1 for snpIdx in SNP_indices]
geno_hdf = h5py.File(genoFile, 'r')
positions = geno_hdf['positions'][:]
pos = [positions[snp] for snp in SNP_indices]
pvalues = r_single.stats.pv20.tolist()

gwas_tuples = list(zip(chrom, pos, pvalues))
gwas_results = pd.DataFrame(gwas_tuples, columns = ['chrom', 'pos', 'pv'])

# plot results
# Manhattan plot
plot.manhattan(gwas_results)
plt = plot.get_pyplot() 
plt.savefig(f'{resultsDir}manhattanPlot_singleLocus.png')
plt.close()

# check if covariate locus is significant
gwas_results[(gwas_results.chrom == covarLocus[0]) & (gwas_results.pos == covarLocus[1])] 

## check for region
covarRange = range(covarLocus[1] - 1000, covarLocus[1] + 1000)

gwas_results_chrom = gwas_results[gwas_results.chrom == covarLocus[0]]

covarRegion = gwas_results_chrom.pos[(gwas_results_chrom.pos > covarLocus[1] - 1000) & (gwas_results_chrom.pos < covarLocus[1] + 1000)]
x = [gwas_results[(gwas_results.chrom == covarLocus[0]) & (gwas_results.pos == pos)] for pos in covarRegion] 


# random trait
random.shuffle(Y)
r_random = scan(G = G, Y= Y, K = K, lik = 'normal', M = None, verbose= True)

chrom = [bisect(chrIdx[:, 1], snpIdx) + 1 for snpIdx in SNP_indices]
geno_hdf = h5py.File(genoFile, 'r')
positions = geno_hdf['positions'][:]
pos = [positions[snp] for snp in SNP_indices]
pvalues = r_random.stats.pv20.tolist()

gwas_tuples = list(zip(chrom, pos, pvalues))
gwas_results = pd.DataFrame(gwas_tuples, columns = ['chrom', 'pos', 'pv'])

# plot results
# Manhattan plot
plot.manhattan(gwas_results)
plt = plot.get_pyplot() 
plt.savefig(f'{resultsDir}manhattanPlot_singleLocus_random.png')
plt.close()

# permute Kinship rows and columns jointly
K_DF = pd.DataFrame(data = K)
K_randomIdx = K_DF.index.values.copy()
random.shuffle(K_randomIdx)
K_DF.index = K_randomIdx  
K_DF.columns = K_randomIdx
K_DF_random = K_DF.sort_index(axis = 0) 
K_DF_random = K_DF_random.sort_index(axis = 1) 
K_random = K_DF_random.to_numpy()

r_K_random = scan(G = G, Y = Y, K = K_random, lik = 'normal', M = None, verbose = True)
chrom = [bisect(chrIdx[:, 1], snpIdx) + 1 for snpIdx in SNP_indices]
geno_hdf = h5py.File(genoFile, 'r')
positions = geno_hdf['positions'][:]
pos = [positions[snp] for snp in SNP_indices]
pvalues = r_K_random.stats.pv20.tolist()

gwas_tuples = list(zip(chrom, pos, pvalues))
gwas_results = pd.DataFrame(gwas_tuples, columns = ['chrom', 'pos', 'pv'])

# plot results
# Manhattan plot
plot.manhattan(gwas_results)
plt = plot.get_pyplot() 
plt.savefig(f'{resultsDir}manhattanPlot_singleLocus_K_random.png')
plt.close()

# no K matrix
r_no_K = scan(G = G, Y = Y, K = None, lik = 'normal', M = None, verbose = True)
chrom = [bisect(chrIdx[:, 1], snpIdx) + 1 for snpIdx in SNP_indices]
geno_hdf = h5py.File(genoFile, 'r')
positions = geno_hdf['positions'][:]
pos = [positions[snp] for snp in SNP_indices]
pvalues = r_no_K.stats.pv20.tolist()

gwas_tuples = list(zip(chrom, pos, pvalues))
gwas_results = pd.DataFrame(gwas_tuples, columns = ['chrom', 'pos', 'pv'])

# plot results
# Manhattan plot
plot.manhattan(gwas_results)
plt = plot.get_pyplot() 
plt.savefig(f'{resultsDir}manhattanPlot_singleLocus_no_K.png')
plt.close()

# principle comnponent correction for population structure
PCs = 5
pca = PCA(n_components=PCs)
pc_out = pca.fit_transform(K)
dfCols = [f'PC-{PC}' for PC in range(1, PCs + 1)] 
pcdf = pd.DataFrame(data = pc_out , columns = dfCols)

r_PC = scan(G = G, Y = Y, K = None, M = pc_out, lik = 'normal', verbose = True)
chrom = [bisect(chrIdx[:, 1], snpIdx) + 1 for snpIdx in SNP_indices]
geno_hdf = h5py.File(genoFile, 'r')
positions = geno_hdf['positions'][:]
pos = [positions[snp] for snp in SNP_indices]
pvalues = r_PC.stats.pv20.tolist()

gwas_tuples = list(zip(chrom, pos, pvalues))
gwas_results = pd.DataFrame(gwas_tuples, columns = ['chrom', 'pos', 'pv'])

# plot results
# Manhattan plot
plot.manhattan(gwas_results)
plt = plot.get_pyplot() 
plt.savefig(f'{resultsDir}manhattanPlot_singleLocus_PC_{PCs}.png')
plt.close()


#TODO: pmake manhattan plots with higher MAF cut-off


'''






