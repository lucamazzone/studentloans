/* Change working directory */
cd "/Users/lucamazzone/iCloud Drive (Archive)/Desktop/Work/Student Loans/NCES";

/* Increase memory size to allow for dataset */
set memory 300m;

/* Clear everything */
clear;

/* Open Stata dataset */
use "els_02_12_byf3pststu_v1_0";

summarize F3FEDCUM3
