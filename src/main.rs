extern crate clap;
extern crate ndarray;
extern crate rayon;

use clap::Parser;
use std::fs::File;
use std::io::{BufReader, Read};
use std::error::Error;
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, s, Axis};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Parser, Debug)] 
#[command(version, about)]
struct Args {
    #[arg(short, long, value_name = "FILE", required = true)]
    aln: String,
}

struct AlnRec {
    qseqid: String,
    sseqid: String,
    slen: usize,
    sstart: usize,
    send: usize,
    bitscore: f32,
}

struct Alignments {
    qseqid_set: HashSet<String>,
    sseqid_set: HashSet<String>,
    slen_map: HashMap<String, usize>,
    sstarts_map: HashMap<String, HashMap<String, usize>>,
    sends_map: HashMap<String, HashMap<String, usize>>,
    q_s_bitscore_map: HashMap<String, HashMap<String, f32>>,
}


fn read_alignment(file_path: &str) -> 
    Result<Alignments, Box<dyn Error>> {
    println!("Attempting to open alignment and autodetect if gzipped or not");
    // Check if the file path ends with ".gz"
    let is_gzipped = file_path.ends_with(".gz");

    // Open the file accordingly
    let file: Box<dyn Read> = if is_gzipped {
        let file = File::open(file_path)?;
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        let file = File::open(file_path)?;
        Box::new(file)
    };

    let reader = BufReader::new(file);

    // Prepare CSV reader
    let mut csv_reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true) // Allow flexible reading of alignments
        .delimiter(b'\t') // Set the delimiter to tab
        .from_reader(reader);

    
    println!("Starting to read in alignments");
    
    // subject lengths as a hashmap
    let mut slen_map: HashMap<String, usize> = HashMap::new();

    // Collect unique qseqids and sseqids
    let mut qseqid_set: HashSet<String> = HashSet::new();
    let mut sseqid_set: HashSet<String> = HashSet::new();

    // Per subject query sstart and sends as nested HashMap[subject_id][query_id] = u32 
    let mut sstarts_map: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut sends_map: HashMap<String, HashMap<String, usize>> = HashMap::new();

    // Bitscores for each subject / query pairing HashMap[Query][Subject] = f32 (bitscore)
    let mut q_s_bitscore_map: HashMap<String, HashMap<String, f32>> = HashMap::new();

    for result in csv_reader.records() { // Iterate over each record
        let record = result?;
        // Extract values from specified columns
        let qseqid = record.get(0).unwrap_or("").to_string();
        let sseqid = record.get(1).unwrap_or("").to_string();
        let slen = record.get(13).unwrap_or("0").parse::<usize>().unwrap_or(0);
        let sstart = record.get(8).unwrap_or("0").parse::<usize>().unwrap_or(0);
        let send = record.get(9).unwrap_or("0").parse::<usize>().unwrap_or(0);
        let bitscore = record.get(11).unwrap_or("0.0").parse::<f32>().unwrap_or(0.0);
        
        qseqid_set.insert(qseqid.clone());
        sseqid_set.insert(sseqid.clone());
        slen_map.insert(sseqid.clone(), slen.clone());
        sstarts_map.entry(
            sseqid.clone()).or_insert(HashMap::new()
        ).insert(qseqid.clone(), sstart.clone());
        sends_map.entry(
            sseqid.clone()).or_insert(HashMap::new()
        ).insert(qseqid.clone(), send.clone());
        q_s_bitscore_map.entry(
            qseqid.clone()).or_insert(HashMap::new()
        ).insert(sseqid.clone(), bitscore.clone());
    }

    println!("Completed reading of alignments.");

    Ok(
        Alignments{
            qseqid_set,
            sseqid_set,
            slen_map,
            sstarts_map,
            sends_map,
            q_s_bitscore_map,
        }
    ) 
}

fn build_subject_cover(
    slen: usize,
    sstarts: Vec<usize>,
    sends: Vec<usize>,
) ->    Array1<u32> {

    let mut cov_arr = Array1::<u32>::zeros(slen);

    for i in 0..sstarts.len() {
        if sends[i] > slen {
            println!("Send {} is greater than slen {}!", sends[i], slen);
        }
        cov_arr.slice_mut(s![sstarts[i]-1..sends[i]]).mapv_inplace(|x| x + 1);
    }

    cov_arr
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let opts = Args::parse();

    let strim_5 = 15;
    let strim_3 = 15;
    let sd_mean_cutoff = 2.0;
    
    let mut alignments = read_alignment(&opts.aln)?;

    println!("Number of Subjects: {:?}", alignments.sseqid_set.len());
    println!("Number of Queries: {:?}", alignments.qseqid_set.len());


    
    // 1.  First subject depth / evenness filter.
    // Thread safe place to add our filtered subjects

    let subjects_to_be_removed: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));

    alignments.sseqid_set.par_iter().for_each(|subj|{
        let mut s_queries: Vec<_> = alignments.qseqid_set.intersection(
            &alignments.sstarts_map[subj].keys().cloned().collect()
        ).cloned().collect();
        s_queries.sort();
        let mut sstarts_vec: Vec<usize> = Vec::new();
        let mut sends_vec: Vec<usize> = Vec::new();
        for s_q in &s_queries {
            sstarts_vec.push(
                alignments.sstarts_map[subj][s_q].clone()
            );
            sends_vec.push(
                alignments.sends_map[subj][s_q].clone()
            );            
        }
        let cov_arr = build_subject_cover(
            alignments.slen_map[subj],
            sstarts_vec,
            sends_vec
        );

        let  cov_arr_f32: Array1<f32>;

        if cov_arr.len() > (strim_5 + strim_3) {
                cov_arr_f32 = cov_arr.slice(s![strim_5..(cov_arr.len() - strim_3)]).mapv(|x| x as f32);
        }  else {
            cov_arr_f32 = cov_arr.mapv(|x| x as f32);
        }
        let cov_mean = cov_arr_f32.mean_axis(Axis(0)).unwrap().into_scalar();
        let cov_std = cov_arr_f32.std_axis(Axis(0), 0.0).into_scalar();
        let cov_pass = (cov_std / cov_mean) < sd_mean_cutoff;
        if !cov_pass {
            // Acquire lock for writing
            let mut subjects_to_be_removed_set = subjects_to_be_removed.lock().unwrap();
            subjects_to_be_removed_set.insert(subj.clone());
        }
    }); // End first coverage pass iteration
     // Acquire lock for reading
    let subjects_to_be_removed_set = subjects_to_be_removed.lock().unwrap();
    alignments.sseqid_set = alignments.sseqid_set.difference(
        &subjects_to_be_removed_set
    ).cloned().collect();
    println!("{:?} subjects pruned in first filter. There are {} subjects remaining", subjects_to_be_removed_set.len(), alignments.sseqid_set.len());



    Ok(())
}
