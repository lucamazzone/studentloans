
clear all
set more off
cd "DIRECTORY"



*********
*table 1*
*********

insheet using aggregate_fy_stocks.csv, clear
keep if full_sample==1
foreach var of varlist tot_bal undergrad_l grad_l parplus_l {
replace `var'=`var'/cpi
}
foreach var of varlist undergrad_l grad_l parplus_l {
replace `var'=`var'/tot_bal
}
foreach var of varlist undergrad_b both grad_b parplus_b {
replace `var'=`var'/borrowers
}
replace undergrad_b=undergrad_b-both
tabstat tot_bal undergrad_l grad_l parplus_l borrowers undergrad_b both grad_b parplus_b if fy>=1982, by(fy) notot


*********
*table 2*
*********
*NCES DATA, see table notes for sources


*********
*table 3*
*********
insheet using aggregate_fy_stocks.csv, clear
replace sel_index=0 if full_sample==1
drop if sel_index==.
keep fy sel_index borrowers tot_bal undergrad_l undergrad_b 
reshape wide borrowers tot_bal undergrad_l undergrad_b , i(fy) j(sel_index)
tabstat borrowers*, by(fy) notot
*latabstat borrowers*, by(fy) notot

*********
*table 4*
*********
tabstat tot_bal*, by(fy) notot
*latabstat tot_bal*, by(fy) notot


*********
*table 5*
*********
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


*********
*table 6*
*********

insheet using repay_outcomes.csv, clear
replace sel_index=0 if full_sample==1
drop if sel_index==.
replace md_tot_bal=round(md_tot_bal/cpi)
keep fy sel_index md_tot_bal 
reshape wide md_tot_bal , i(fy) j(sel_index)
tabstat md_tot_bal*, by(fy) notot

*********
*table 7*
*********

insheet using labormarket_outcomes.csv, clear
replace sel_index=0 if full_sample==1
gen de=md_debt_service/md_earnings
keep if sel_index~=.
keep if year==2
keep sel_index md_de de repay_y
reshape wide md_de de ,i(repay_) j(sel_)
format de* %9.2f
tabstat de*, by(repay_y) notot

*********
*table 8*
*********

insheet using repay_outcomes.csv, clear
drop if fy<1972|fy>2012
replace sel_index=0 if full_sample==1
drop if sel_index==.
keep fy sel_index alt_cdr3 alt_cdr5 rr2 neg_am2 
replace rr=1-rr
reshape wide alt_cdr3 alt_cdr5 rr2 neg_am2 , i(fy) j(sel_index)
tabstat neg_am2* alt_cdr5*, by(fy) notot

*********
*table 9*
*********
/*
cd "DIRECTORY"

use if repay_y==2000|repay_y==2011 using master_stab_flows.dta, clear

gen in_default=default_y<=repay_y+2
keep if (fy==repay_y)|(fy==repay_y+2)
gen e=f2.earnings/f2.cpi
replace earnings=e
gen neg_am=f2.tot_bal>=tot_bal
replace neg_am=. if f2.tot_bal==.

keep if fy==repay_y
drop if tot_bal/cpi<50
replace tot_bal=tot_bal/cpi
replace grad_l=grad_l/cpi
replace fam_inc=fam_inc/cpi
drop if sch_code==.
sort sch_code

gen unemp=e==0
replace unemp=. if e==.
drop n
gen n=1/1000
replace depend=0 if depend==2
replace tot_bal=tot_bal/1000
replace grad_l=grad_l/1000
gen d_fam_inc=fam_inc*(1-depend)

gen grad=grad_l>0
gen duration=repay_y-entry
replace duration=4 if duration>4

gen complete=0
replace complete=1 if grad_2_yr=="Y"&typen==1
replace complete=1 if grad_2_yr=="Y"&typen==3
replace complete=1 if grad_2_yr=="Y"&typen==5

replace complete=1 if grad_4_yr=="Y"&typen==2
replace complete=1 if grad_4_yr=="Y"&typen==4
replace complete=1 if grad_4_yr=="Y"&typen==6

gen earnaa=0
replace earnaa=1 if grad_2_yr=="Y"
gen earnba=0
replace earnba=1 if grad_4_yr=="Y"
gen nodegree=0
replace nodegree=1 if grad_2=="N"&grad_4=="N"

gen drop=0
replace drop=1 if typen==1&grad_2_yr=="N"
replace drop=1 if typen==3&grad_2_yr=="N"
replace drop=1 if typen==5&grad_2_yr=="N"
replace drop=1 if typen==2&grad_4_yr=="N"
replace drop=1 if typen==4&grad_4_yr=="N"
replace drop=1 if typen==6&grad_4_yr=="N"

replace pell_amt=0 if pell_amt==.

tab f_typen, gen(typen)

tab f_sel_index, gen(type)

 label variable type1 "For Profit"
 label variable type2 "2-Year"
 label variable type3 "Least Selective"
 label variable type4 "Somewhat Selective"
 label variable type5 "Selective"
 label variable type6 "Graduate Only"

bys repay_y sch_code: egen i=mean(int_rt)
bys repay_y sel_index: egen ii=mean(int_rt)
bys repay_y : egen iii=mean(int_rt)
replace int_rt=i if int_rt==.
replace int_rt=ii if int_rt==.
replace int_rt=iii if int_rt==.

gen debt_service=tot_bal*(int_rt+int_rt/((1+int_rt)^10-1))*1000
gen de=debt_service/e
replace de=10 if e==0
renam de debt_earnings
*windsorize
replace debt_earnings=.0102554 if debt_earnings<.0102554
replace debt_earnings=1.5 if debt_earnings>1.5

gen kids=0
replace kids=1 if dep_ch>=1

*rescale
replace fam_incm=fam_incm/10000
replace fam_inc=0 if fam_inc<0
replace d_fam_inc=d_fam_inc/10000
replace d_fam_inc=0 if d_fam_inc<0

replace earnings=earnings/10000

keep if fam_inc!=.
gen age=fy-yob
gen married=0
replace married=1 if fil_stat==2
 
*label
label variable grad "Graduate Student"
label variable debt_earnings "Debt/Earnings"
label variable earnings "Earnings"
label variable duration "Duration"
label variable pell_amt "Pell Eligible"
label variable unemp "No Earnings"
label variable fam_inc "Family Income"
label variable d_fam_inc "Independent Family Income"
label variable kids "Has Children"
label variable earnaa "Attained AA"
label variable earnba "Attained BA"
label variable drop "Dropout"
label variable married "Married"
label variable age "Age"

global labor earnings unemp 
global background fam_inc depend pell_amt
global  duration drop
global loan debt_earnings 

global school type1 type2 type3 type4 type5 
global type typen1 typen2 typen3 typen4 typen5
global individual grad age  depend debt_earnings  earnings duration tot_bal repay_bal_int ever_pell exp_fam_contrib  debt_service unemp fam_incm d_fam_inc earnaa earnba married drop kids
global pre grad depend fam_inc  d_fam_inc  age   pell_amt  
global post drop duration  earnings  debt_earnings tot_bal 
global borrow debt_earnings int_rt tot_bal

cd "G:\%Office of Tax Analysis\_Individual Staff\Nick_Turner\Shared_Data\STAB\Constantine\data_wal"

oaxaca  in_def $school $individual , by(fy) logit relax


eststo clear

eststo: reg in_def $school, robust

eststo: reg in_def $school  $pre   , robust

eststo: reg in_def $school  $individual, robust

eststo: reg in_def $school  $pre  if fy==2000, robust

eststo: reg in_def $school  $pre    if fy==2011, robust

eststo: reg in_def $school  $individual  if fy==2000, robust

eststo: reg in_def $school  $individual if fy==2011, robust

esttab using regressions_old.csv, label nostar  r2 replace cells(b(star fmt(3)) se(par fmt(4))) keep( $school $individual  ) 

eststo clear

logit in_def $school, robust iterate(100)
eststo: mfx

logit in_def $school  $pre   , robust iterate(100)
eststo: mfx

logit in_def $school  $individual, robust iterate(100)
eststo: mfx

logit in_def $school  $pre  if fy==2000, robust iterate(100)
eststo: mfx

logit in_def $school  $pre    if fy==2011, robust iterate(100)
eststo: mfx

logit in_def $school  $individual  if fy==2000, robust iterate(100)
eststo: mfx

logit in_def $school  $individual if fy==2011, robust
eststo: mfx

esttab using regressions.csv, label margin nostar  r2 replace cells(b(star fmt(3)) se(par fmt(4))) keep( $school $individual  ) 



**********
*table 10*
**********


eststo clear
eststo: oaxaca  in_def $school , by(fy) logit relax
oaxaca  in_def $school , by(fy) logit relax swap
eststo: oaxaca  in_def $school $pre , by(fy) logit relax
oaxaca  in_def $school $pre , by(fy) logit relax swap
eststo: oaxaca  in_def $school $type $individual    , by(fy) logit relax
eststo: oaxaca  in_def  $individual    if sel_index==1, by(fy) logit relax
eststo: oaxaca  in_def  $individual    if sel_index==2, by(fy) logit relax
eststo: oaxaca  in_def  $individual    if sel_index==3, by(fy) logit relax
eststo: oaxaca  in_def  $individual    if sel_index==4, by(fy) logit relax
eststo: oaxaca  in_def  $individual    if sel_index==5, by(fy) logit relax
eststo: oaxaca  in_def  $individual    if sel_index==6, by(fy) logit relax

esttab using oaxaca_alt.csv, label nostar replace se keep(difference  endowments $school $type $individual  ) 

*/
cd "DIRECTORY"

**********
*figure 1*
**********
insheet using repay_outcomes.csv, clear 
keep if full==1
keep fy i_cdr2
gen official=.
replace official=	0.176	 if fy==	1987
replace official=	0.172	 if fy==	1988
replace official=	0.214	 if fy==	1989
replace official=	0.224	 if fy==	1990
replace official=	0.178	 if fy==	1991
replace official=	0.15	 if fy==	1992
replace official=	0.107	 if fy==	1993
replace official=	0.106	 if fy==	1994
replace official=	0.104	 if fy==	1995
replace official=	0.096	 if fy==	1996
replace official=	0.088	 if fy==	1997
replace official=	0.069	 if fy==	1998
replace official=	0.056	 if fy==	1999
replace official=	0.059	 if fy==	2000
replace official=	0.054	 if fy==	2001
replace official=	0.052	 if fy==	2002
replace official=	0.045	 if fy==	2003
replace official=	0.051	 if fy==	2004
replace official=	0.046	 if fy==	2005
replace official=	0.052	 if fy==	2006
replace official=	0.067	 if fy==	2007
replace official=	0.07	 if fy==	2008
replace official=	0.088	 if fy==	2009
replace official=	0.091	 if fy==	2010
replace official=	0.1	 if fy==	2011
label variable fy " "
label variable official "ED"
label variable i_cdr "Treasury"
graph twoway line i_cdr official fy if fy>1972, scheme(s1mono) lwidth(thick thick thick thick thick thick) lcolor(navy ebblue ebg edkblue ltblue emidblue eltblue blue*1.5)  ylabel(,grid)
graph export figure_1_defr.pdf, replace


**********
*figure 2*
**********
insheet using first_year_new_volume_all.csv, clear 
drop if sel_index==.
keep fy sel_index new_borrowers
reshape wide new_borrowers,i(fy) j(sel_index)
gen x1=new_borrowers6
gen x2=x1+new_borrowers5
gen x3=x2+new_borrowers4
gen x4=x3+new_borrowers3
gen x5=x4+new_borrowers2
gen x6=x5+new_borrowers1
drop if fy<1982
label variable fy " "
twoway area x6 fy, col(ebg) sort || rarea x5 x6 fy, col(navy)  sort || rarea x4 x5 fy, col(edkblue) sort || rarea x3 x4 fy, col(eltblue) sort || rarea x2 x3 fy, col(emidblue) sort || rarea x1 x2 fy, col(midblue*.5) sort   legend(label(1 "Graduate Only") label(2 "For Profit") label(3 "2-Year Pub/Priv") label(4 "Non-Selective") label(5 "Somewhat Selective") label(6 "Selective")  order( 2 3 4 5 6 1),  ) scheme(s1mono)  ytitle("Borrowers (thousands)", size(small)) ylabel(,labsize(small) ) ylabel(,grid) xscale(range(1982 2014)) xlabel(1982(4)2014)
graph export figure_2_entry_by_sel_type.png, replace

insheet using repay_outcomes.csv, clear 
drop if sel_index==.
keep fy sel_index borrowers
reshape wide borrowers,i(fy) j(sel_index)
gen x1=borrowers6
gen x2=x1+borrowers5
gen x3=x2+borrowers4
gen x4=x3+borrowers3
gen x5=x4+borrowers2
gen x6=x5+borrowers1
drop if fy<1982
label variable fy " "
twoway area x6 fy, col(ebg) sort || rarea x5 x6 fy, col(navy)  sort || rarea x4 x5 fy, col(edkblue) sort || rarea x3 x4 fy, col(eltblue) sort || rarea x2 x3 fy, col(emidblue) sort || rarea x1 x2 fy, col(midblue*.5) sort   legend(label(1 "Graduate Only") label(2 "For Profit") label(3 "2-Year Pub/Priv") label(4 "Non-Selective") label(5 "Somewhat Selective") label(6 "Selective") order( 2 3 4 5 6 1),  ) scheme(s1mono)  ytitle("Borrowers (thousands)", size(small)) ylabel(,labsize(small) ) ylabel(,grid) xscale(range(1982 2014)) xlabel(1982(4)2014)
graph export figure_2b_repay_by_sel_type.png, replace


**********
*figure 3*
**********

insheet using repay_outcomes.csv, clear 
keep fy sel_index borrowers tot_bal
drop if sel_index==.
reshape wide borrowers tot_bal ,i(sel_index) j(fy)
graph bar borrowers2000 borrowers2011, over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)) )	 scheme(s1mono) legend(label(1 "2000 Cohort") label(2 "2011 Cohort"))  ytitle("Borrowers (thousands)", size(small)) ylabel(,labsize(small) grid) bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) 
graph export figure_3_repayment.png, replace


**********
*figure 4*
**********

insheet using repay_outcomes.csv, clear 
drop if sel_index==.|sel_index==6
keep if fy==2011

keep fy sel_index md_dep_fam_inc md_ind_fam_inc dependent grad2 grad4 age_entry pct_black unemp_rate poverty_rate median_hh first_gen
reshape wide md_dep_fam_inc md_ind_fam_inc dependent grad2 grad4 age_entry pct_black  unemp_rate poverty_rate median_hh first_gen,i(sel_index) j(fy)

replace dependent2011=dependent2011*100
replace grad22011=grad22011*100
replace grad42011=grad42011*100
replace first_gen2011=first_gen2011*100
replace poverty_rate2011=poverty_rate2011*100

graph bar md_dep_fam_inc2011, over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)) ) scheme(s1mono)  bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy))	  legend(label(1 "2000 Cohort") label(2 "2012 Cohort")) title("Parent's Income")  ytitle(" ", size(small)) ylabel(,labsize(small)) saving(family, replace)
graph bar age_entry2011, over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)) )	scheme(s1mono)  bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) legend(label(1 "2-Year") label(2 "4-Yr")) title("Median Entry Age")  ytitle("%", size(small)) ylabel(,labsize(small)) saving(age, replace)
graph bar dependent2011, over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)) )	scheme(s1mono)  bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) legend(label(1 "2000 Cohort") label(2 "2012 Cohort")) title("% Dependent")  ytitle("%", size(small)) ylabel(,labsize(small)) saving(dependent, replace)
graph bar first_gen2011, over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)) )	scheme(s1mono) bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy))  legend(label(1 "2000 Cohort") label(2 "2012 Cohort")) title("% First Generation")  ytitle("%", size(small)) ylabel(,labsize(small)) saving(first_gen, replace)
graph bar poverty_rate2011, over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)) )	scheme(s1mono)  bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) legend(label(1 "2000 Cohort") label(2 "2012 Cohort")) title("Local Poverty Rate")  ytitle("%", size(small)) ylabel(,labsize(small)) saving(poverty, replace)
graph bar grad22011 grad42011, over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)) )	scheme(s1mono)  bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) legend(label(1 "2-Year") label(2 "4-Yr")) title("Completion Rates")  ytitle("%", size(small)) ylabel(,labsize(small)) saving(grad, replace)
gr combine family.gph age.gph  dependent.gph first_gen.gph poverty.gph grad.gph 
graph export figure_4_borrower_characteristics.pdf, replace 

**********
*figure 5*
**********

insheet using labormarket_outcomes.csv, clear
keep if sel_index~=.
keep if year==2
keep sel_index md_earnings unemp repay_y
reshape wide md_earnings unemp ,i(sel_index) j(repay_)

graph bar unemployed2000 unemployed2011 , over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)))	 scheme(s1mono) legend(label(1 "2000 Cohort") label(2 "2011 Cohort"))  ylabel(,labsize(small) grid) bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) 
graph export figure_8_unemployed.png, replace

graph bar md_earnings2000 md_earnings2011 , over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)))	 scheme(s1mono) legend(label(1 "2000 Cohort") label(2 "2011 Cohort"))  ylabel(,labsize(small) grid) bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) 
graph export figure_7_md_earnings.png, replace


**********
*figure 6*
**********
/*
cd "directory"

clear all
use stab_flows.dta, clear
gen cohort_2_yr=fy-2
keep if tot_bal>0
keep if tot_inc>-1
keep if tot_inc<500000
gen I28=0
replace I28=1 if age>24&age<35
gen ln = exp(10.1986+ .989679 * invnorm(uniform()))
graph twoway  (kdensity ln if I28==1&fy==2008&ln<250000, lwidth(thick) lcolor(navy*1.75) ) (kdensity tot_inc if I28==1&fy==2008&tot_inc<250000&tot_bal<25000, scheme(s1mono) lwidth(thick) lcolor(blue*.25))  (kdensity tot_inc if I28==1&fy==2008&tot_inc<250000&tot_bal>25000, lwidth(thick) lcolor(blue*.5)) (kdensity tot_inc if I28==1&fy==2008&tot_inc<250000&tot_bal>75000, lwidth(thick) lcolor(blue*1))  , legend(label(1 "All") label(2 "<25k") label(3 ">25k") label(4 ">75k")) ytitle("Income Density") xtitle("Income") 

cd "G:\%Office of Tax Analysis\_Individual Staff\Nick_Turner\Shared_Data\STAB\Looney update\"

use if repay_y==2000|repay_y==2011 using master_stab_flows.dta, clear
gen e=f2.earnings/f2.cpi
keep if fy==repay_y
replace tot_bal=tot_bal/cpi
replace earnings=e
gen inc_group=.
replace inc_group=1 if earnings<=10000
replace inc_group=2 if earnings>10000&earnings<=20000
replace inc_group=3 if earnings>20000&earnings<=30000
replace inc_group=4 if earnings>30000&earnings<=40000
replace inc_group=5 if earnings>40000&earnings<=50000
replace inc_group=6 if earnings>50000&earnings<=60000
replace inc_group=7 if earnings>60000&earnings<=70000
replace inc_group=8 if earnings>70000&earnings<=80000
replace inc_group=9 if earnings>80000&earnings<=90000
replace inc_group=10 if earnings>90000&earnings<=100000
replace inc_group=11 if earnings>=100000&earnings!=.
gen bor_group=.
replace bor_group=1 if tot_bal>=0&tot_bal<15000
replace bor_group=2 if tot_bal>=15000&tot_bal<25000
replace bor_group=3 if tot_bal>=25000&tot_bal<50000
replace bor_group=4 if tot_bal>=50000
gen n=1
collapse (sum) n (mean) tot_bal earnings, by(repay_y inc_group bor_group)
graph bar earnings if repay_y==2011, over(bor_group) scheme(s1mono) 

*/


**********
*figure 7*
**********

insheet using repay_outcomes.csv, clear 
drop if sel_index==.
keep fy alt_cdr3 sel_index
reshape wide alt_cdr3,i(fy) j(sel_index)
label variable fy " "
label variable alt_cdr31  "For-Profit"
label variable alt_cdr32  "2-Year"
label variable alt_cdr33  "Non-Selective"
label variable alt_cdr34  "Somewhat Selective"
label variable alt_cdr35  "Selective"
label variable alt_cdr36  "Grad Only"

graph twoway line alt* fy if fy>1972&fy<=2012, scheme(s1mono) lwidth(thick thick thick thick thick thick) lcolor(navy  edkblue midblue*.5 emidblue eltblue  ebg)  ylabel(,grid) xlabel(1972 (10) 2010)
graph export figure_9_cdr3.pdf, replace


**********
*figure 8*
**********

insheet using repay_outcomes.csv, clear 
drop if sel_index==.
keep fy i_cdr3 sel_index
reshape wide i_cdr,i(sel) j(fy)
graph bar i_cdr32000 i_cdr32011 , over(sel_index, rel(1 "For-profit" 2 "2-Year" 3 "Non-Selective" 4 "Somewhat Selective" 5 "Selective"  6 "Graduate Only") lab(alt labs(small)))	 scheme(s1mono) legend(label(1 "2000 Cohort") label(2 "2011 Cohort"))  ylabel(,labsize(small) grid) bar(1, color(ebblue) lcolor(black)) bar(2, lcolor(black) color(navy)) 
graph export figure_10_cdr3.png, replace

**********
*figure 9*
**********

insheet using flows_of_borrowers.csv,clear
keep if full==1
gen net_change=new_borrower-paid_off
label variable net_change "Net Change in Borrowers"
label variable new_borrower "Inflows: New Borrowers"
label variable paid_off "Outflows: Paid Off Loans"
label variable fy " "
tabstat new_borrower entered_repay paid_off dur_repay paid_length , by(fy)
graph twoway line new_borrower paid_off net_change fy if fy>1990, scheme(s1mono) lwidth(thick thick thick) lcolor(navy ebblue maroon) ylabel(,labsize(small) grid)
graph export figure_14_flows.pdf, replace

