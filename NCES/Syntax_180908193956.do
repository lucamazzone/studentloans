/************************************************************************
*** You may need to edit this code.                                  ***
***                                                                  ***
*** Please check all CD statements and USE statements before         ***
*** running this code.                                               ***
***                                                                  ***
*** You may have selected variables that contain missing data or     ***
*** valid skips. You may wish to recode one or both of these special ***
*** values. You need to consult the Variable Description to see if   ***
*** these special codes apply to your extracted variables. You can   ***
*** recode these special values to missing using the following       ***
*** sample code:                                                     ***
***                                                                  ***
*** replace {variable name} = . if {variable name} = {value};        ***
***                                                                  ***
*** Replace {variable name} above with the name of the variable you  ***
*** wish to recode. Replace {value} with the special value you wish  ***
*** to recode to missing.                                            ***
***                                                                  ***
*** It is important to retain full sample weights, replicate         ***
*** weights, and identification numbers as appropriate.              ***
************************************************************************/

/* Change delimiter to a semi-colon */
#delimit;

/* Change working directory */
cd "C:\EDAT\ELS\";

/* Increase memory size to allow for dataset */
set memory 300m;

/* Clear everything */
clear;

/* Open Stata dataset */
use "els_02_12_byf3pststu_v1_0";

/* Keep only selected variables */
keep
   F3FEDLNCUM
   F3FEDLNDUE 
   F3FEDCUM3
   F3FEDDUE3
   F3SUBCUM3
   F3STFCUM3
   F3CNSDUE   
   F3CNSOWED  
   F3PELLCUM
   F3PLUSCUM
   F3GPLUSCUM
   F3PERKCUM  
   F3STAFSCUM
   F3STAFTCUM
   F3STAFUCUM 
   F3SPPCUM   
   F3SPPDUE   
   F3SPPOWED  
   F3SSPCUM  
   F3STPCUM 
   F3STPDUE
   F3STPOWED  
   F3TOTOWED
   F3PELLYRS
   F3PELL0405 
   F3PELL0506 
   F3PELL0607
   F3PELL0708 
   F3PELL0809 
   F3PELL0910 
   F3PELL1011 
   F3PELL1112
   F3PELL1213 
   F3PLUSYRS
   F3PLUS0405
   F3PLUS0506 
   F3PLUS0607 
   F3PLUS0708 
   F3PLUS0809
   F3PLUS0910
   F3PLUS1011 
   F3PLUS1112 
   F3PLUS1213 
   F3STAFYRS  
   F3STFY0405
   F3STFY0506 
   F3STFY0607 
   F3STFY0708 
   F3STFY0809 
   F3STFY0910
   F3STFY1011 
   F3STFY1112 
   F3STFY1213 
   F3STSB0405 
   F3STSB0506 
   F3STSB0607 
   F3STSB0708
   F3STSB0809 
   F3STSB0910 
   F3STSB1011 
   F3STSB1112 
   F3STSB1213
   F3STUN0405 
   F3STUN0506 
   F3STUN0607 
   F3STUN0708 
   F3STUN0809
   F3STUN0910 
   F3STUN1011 
   F3STUN1112 
   F3STUN1213
   F2B01
   F2B02
   F2B03
   F2B03_P
   F2B04
   F2B05A
   F2B05B
   F2B05C
   F2B05D
   F2B05E
   F2B05F
   F2B05G
   F2B06
   F2B07
   F2B08A
   F2B08B
   F2B08C
   F2B08D
   F2B08E
   F2B08F
   F2B08G
   F2B08H
   F2B08I
   F2B08J
   F2B08K
   F2B08L
   F2B08N
   F2B08NA
   F2B09
   F2B10
   F2B10_P
   F2B11A
   F2B11B
   F2B11C
   F2B11D
   F2B11E
   F2B11F
   F2B11G
   F2B11H
   F2B11I
   F2B11J
   F2B11K
   F2B11L
   F2B11N
   F2B11NA
   F2B12
   F2B13A
   F2B13B
   F2B13C
   F2B13D
   F2B13E
   F2B13F
   F2B14
   F2B15
   F2B16A
   F2B16B
   F2B16C
   F2B17A
   F2B17B
   F2B17C
   F2B17D
   F2B18A
   F2B18B
   F2B18C
   F2B18D
   F2B18E
   F2B18F
   F2B18G
   F2B19A
   F2B19B
   F2B19C
   F2B19D
   F2B19E
   F2B19F
   F2B19G
   F2B19H
   F2B19I
   F2B19J
   F2B19K
   F2B20A
   F2B20B
   F2B20C
   F2B20D
   F2B20E
   F2B20F
   F2B20G
   F2B20H
   F2B21A
   F2B21B
   F2B21C
   F2B21D
   F2B21E
   F2B21F
   F2B21G
   F2B21H
   F2B21I
   F2B21J
   F2B21K
   F2B21L
   F2B22
   F2B23A
   F2B24
   F2B25A
   F2B25B
   F2B25C
   F2B25D
   F2B25E
   F2B25F
   F2B25G
   F2B25H
   F2B26R
   F2B26P
   F2B27
   F2B28R
   F2B28P
   F2B29A
   F2B29B
   F2B29C
   F2B29D
   F2B29E
   F2B29F
   F2B29G
   F2B29H
   F2B29I
   F2B29J
   F2B29K
   F2B30
   STU_ID
   SCH_ID
   STRAT_ID
   PSU
   F1SCH_ID
   F1UNIV1
   F1UNIV2A
   F1UNIV2B
   F2UNIV1
   F2UNIV_P
   F3UNIV
   F3UNIVG10
   F3UNIVG12
   G10COHRT
   G12COHRT
   BYSTUWT
   BYEXPWT
   F1QWT
   F1PNLWT
   F1EXPWT
   F1XPNLWT
   F1TRSCWT
   F2QTSCWT
   F2QWT
   F2F1WT
   F2BYWT
   F3QWT
   F3BYPNLWT
   F3F1PNLWT
   F3QTSCWT
   F3BYTSCWT
   F3F1TSCWT
   F3QTSCWT_O
   F3BYTSCWT_O
   F3F1TSCWT_O
   PSWT
   F3BYPNLPSWT
   F3BYTSCPSWT
   F3F1PNLPSWT
   F3F1TSCPSWT
   F3QPSWT
   F3QTSCPSWT
   PSTSCWT
   ;

/* Compress the data to save space */
compress;

/* Save dataset */
save "els_02_12_byf3pststu_v1_0_180908193956", replace;

/* Display frequencies for the categorical variables */
tabulate F3PELLYRS;
tabulate F2B01;
tabulate F2B02;
tabulate F2B03_P;
tabulate F2B04;
tabulate F2B05A;
tabulate F2B05B;
tabulate F2B05C;
tabulate F2B05D;
tabulate F2B05E;
tabulate F2B05F;
tabulate F2B05G;
tabulate F2B06;
tabulate F2B07;
tabulate F2B08A;
tabulate F2B08B;
tabulate F2B08C;
tabulate F2B08D;
tabulate F2B08E;
tabulate F2B08F;
tabulate F2B08G;
tabulate F2B08H;
tabulate F2B08I;
tabulate F2B08J;
tabulate F2B08K;
tabulate F2B08L;
tabulate F2B08N;
tabulate F2B09;
tabulate F2B10_P;
tabulate F2B11A;
tabulate F2B11B;
tabulate F2B11C;
tabulate F2B11D;
tabulate F2B11E;
tabulate F2B11F;
tabulate F2B11G;
tabulate F2B11H;
tabulate F2B11I;
tabulate F2B11J;
tabulate F2B11K;
tabulate F2B11L;
tabulate F2B11N;
tabulate F2B12;
tabulate F2B13A;
tabulate F2B13B;
tabulate F2B13C;
tabulate F2B13D;
tabulate F2B13E;
tabulate F2B13F;
tabulate F2B14;
tabulate F2B15;
tabulate F2B16A;
tabulate F2B16B;
tabulate F2B16C;
tabulate F2B17A;
tabulate F2B17B;
tabulate F2B17C;
tabulate F2B17D;
tabulate F2B18A;
tabulate F2B18B;
tabulate F2B18C;
tabulate F2B18D;
tabulate F2B18E;
tabulate F2B18F;
tabulate F2B18G;
tabulate F2B19A;
tabulate F2B19B;
tabulate F2B19C;
tabulate F2B19D;
tabulate F2B19E;
tabulate F2B19F;
tabulate F2B19G;
tabulate F2B19H;
tabulate F2B19I;
tabulate F2B19J;
tabulate F2B19K;
tabulate F2B20A;
tabulate F2B20B;
tabulate F2B20C;
tabulate F2B20D;
tabulate F2B20E;
tabulate F2B20F;
tabulate F2B20G;
tabulate F2B20H;
tabulate F2B21A;
tabulate F2B21B;
tabulate F2B21C;
tabulate F2B21D;
tabulate F2B21E;
tabulate F2B21F;
tabulate F2B21G;
tabulate F2B21H;
tabulate F2B21I;
tabulate F2B21J;
tabulate F2B21K;
tabulate F2B21L;
tabulate F2B22;
tabulate F2B25A;
tabulate F2B25B;
tabulate F2B25C;
tabulate F2B25D;
tabulate F2B25E;
tabulate F2B25F;
tabulate F2B25G;
tabulate F2B25H;
tabulate F2B26P;
tabulate F2B27;
tabulate F2B28P;
tabulate F2B29A;
tabulate F2B29B;
tabulate F2B29C;
tabulate F2B29D;
tabulate F2B29E;
tabulate F2B29F;
tabulate F2B29G;
tabulate F2B29H;
tabulate F2B29I;
tabulate F2B29J;
tabulate F2B29K;
tabulate F2B30;
/* Display descriptives for the continuous variables */
summarize
      F3FEDLNCUM
      F3FEDLNDUE 
      F3FEDCUM3
      F3FEDDUE3
      F3SUBCUM3
      F3STFCUM3
      F3CNSDUE   
      F3CNSOWED  
      F3PELLCUM
      F3PLUSCUM
      F3GPLUSCUM
      F3PERKCUM  
      F3STAFSCUM
      F3STAFTCUM
      F3STAFUCUM 
      F3SPPCUM   
      F3SPPDUE   
      F3SPPOWED  
      F3SSPCUM  
      F3STPCUM 
      F3STPDUE
      F3STPOWED  
      F3TOTOWED
      F3PELL0405 
      F3PELL0506 
      F3PELL0607
      F3PELL0708 
      F3PELL0809 
      F3PELL0910 
      F3PELL1011 
      F3PELL1112
      F3PELL1213 
      F3PLUSYRS
      F3PLUS0405
      F3PLUS0506 
      F3PLUS0607 
      F3PLUS0708 
      F3PLUS0809
      F3PLUS0910
      F3PLUS1011 
      F3PLUS1112 
      F3PLUS1213 
      F3STAFYRS  
      F3STFY0405
      F3STFY0506 
      F3STFY0607 
      F3STFY0708 
      F3STFY0809 
      F3STFY0910
      F3STFY1011 
      F3STFY1112 
      F3STFY1213 
      F3STSB0405 
      F3STSB0506 
      F3STSB0607 
      F3STSB0708
      F3STSB0809 
      F3STSB0910 
      F3STSB1011 
      F3STSB1112 
      F3STSB1213
      F3STUN0405 
      F3STUN0506 
      F3STUN0607 
      F3STUN0708 
      F3STUN0809
      F3STUN0910 
      F3STUN1011 
      F3STUN1112 
      F3STUN1213
      F2B03
      F2B08NA
      F2B10
      F2B11NA
      F2B23A
      F2B24
      F2B26R
      F2B28R
      STU_ID
      SCH_ID
      STRAT_ID
      F1SCH_ID
      ;

