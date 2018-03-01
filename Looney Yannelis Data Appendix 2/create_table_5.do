
clear all
set more off
cd "DIRECTORY"

insheet using institution_balances.csv, clear
keep if fy==2009
keep alt_cdr5 fract_repaid instnm
save temp,replace
insheet using institution_balances.csv, clear
keep if fy==2000|fy==2014
keep n_borrowers tot_bal instn fy
reshape wide n_borrowers tot_bal ,i(instnm) j(fy)
merge 1:1 instnm using temp, nogen
gsort -tot_bal2000
l instnm tot_bal2000 n_borrowers2000 in 1/25
gsort -tot_bal2014 
l instnm tot_bal2014 n_borrowers2014 alt_cdr5 fract_repaid in 1/25

