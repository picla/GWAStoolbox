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



# Parameters
parser = argparse.ArgumentParser(description = 'Parse parameters for multilocus GWAS')
parser.add_argument('-p', '--phenotype', help = 'Path to phenotype file. CSV file with accession ID in forst column, phenptype values in second column. Filename is used as phenotype name.', required = True)
parser.add_argument('-g', '--genotype', help = 'Path to genotype directory.This directory should contain boht the SNP and kinship matrix. Versions are the same as used for PyGWAS.', required = True)
parser.add_argument('-m', '--maf', help = 'Specify the minor allele frequecny cut-off. Default is set to 0.05', default = 0.05)
parser.add_argument('-o', '--outDir', help = 'Specify the output directory. All results will be saved in this directory.', required = True)
args = parser.parse_args()

# Phenotype (Y)
pheno = pd.read_csv(args.phenotype, index_col = 0)
traits =  pheno.columns.values
# TODO: test if limix can handle NAs. if so, make this optional
# remove NA values
pheno = pheno.dropna(axis = 0, how = 'any') 
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


### MTMM TESTS ###
A = np.matrix('0 1; 1 0')
A0 = np.ones((len(traits), 1)) 
A1 = np.eye(len(traits))
# M = np.repeat(1, Y.shape[0]) 

r = scan(G, Y, K = K, lik = 'normal', A = A, A0 = A0, A1 = A1, verbose = True)




# save results
# link chromosome and positions to p-values and effect sizes
geno_hdf = h5py.File(genoFile, 'r')
chrIdx = geno_hdf['positions'].attrs['chr_regions']
chrom = [bisect(chrIdx[:, 1], snpIdx) + 1 for snpIdx in SNP_indices]
positions = geno_hdf['positions'][:]
pos = [positions[snp] for snp in SNP_indices]

# G effect only
pv10 = r.stats.pv10.tolist()
# G + GxE
pv20 = r.stats.pv20.tolist()
# GxE effect only
pv21 = r.stats.pv21.tolist()

# translate tests
tests = pd.DataFrame(data = ['G', 'G+GxE', 'GxE'], index = ['pv10', 'pv20', 'pv21'], columns = ['test'])

gwas_tuples = list(zip(chrom, pos, pv10, pv20, pv21))
gwas_results = pd.DataFrame(gwas_tuples, columns = ['chrom', 'pos', 'pv10', 'pv20', 'pv21'])

for pv in ['pv10', 'pv20', 'pv21']:
    test = tests.test[pv]
    # plot results
    # Manhattan plot
    gwas_pv = gwas_results[['chrom', 'pos', pv]]
    gwas_pv.columns = ['chrom', 'pos', 'pv'] 
    plot.manhattan(gwas_pv)
    plt = plot.get_pyplot() 
    plt.savefig(f"{args.outDir}/manhattanPlot_{'_'.join(traits)}_{args.maf}_{test}.png")
    plt.close()

    # QQ-plot
    plot.qqplot(gwas_pv.pv)
    plt = plot.get_pyplot()
    plt.savefig(f"{args.outDir}/qqPlot_{'_'.join(traits)}_{args.maf}_{test}.png")
    plt.close()

    # save results
    gwas_pv['maf'] = SNPs_MAF
    gwas_pv['mac'] = gwas_pv.maf * len(acn_indices)
    gwas_pv.mac = gwas_pv.mac.astype(int)
    gwas_pv['GVE'] = np.nan
    gwas_pv.columns.values[0] = 'chr'
    gwas_pv.columns.values[2] = 'pvalue'
    gwas_pv.to_csv(f"{args.outDir}/{'_'.join(traits)}_{args.maf}_MTMM_{test}.csv", index = False)


