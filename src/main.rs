extern crate clap;
extern crate sprs;
extern crate ndarray;

use clap::Parser;
use std::fs::File;
use std::io::{BufReader, Read};
use std::error::Error;
use sprs::{CsMat, TriMat};
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, s, Axis};

#[derive(Parser, Debug)] 
#[command(version, about)]
struct Args {
    #[arg(short, long, value_name = "FILE", required = true)]
    aln: String,
}

#[derive(Debug)]
struct AlnRec {
    qseqid: String,
    sseqid: String,
    slen: usize,
    sstart: usize,
    send: usize,
    bitscore: f32,
}

fn read_alignment(file_path: &str) -> 
    Result<(
        Vec<String>, // queries
        Vec<String>, // subjects
        CsMat<i8>, // Aligned
        CsMat<f32>, // Bitscore
        CsMat<usize>, // sstart
        CsMat<usize>, // send
        HashMap<String, usize>, // subject lengths
        HashMap<String, usize>, // qseqids
        HashMap<String, usize>), // sseqids
        Box<dyn Error>> {
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

    // Vector to store alignments
    let mut alignments: Vec<AlnRec> = Vec::new();

    // Iterate over each record and print selected columns
    for result in csv_reader.records() {
        let record = result?;
        // Extract values from specified columns
        let qseqid = record.get(0).unwrap_or("").to_string();
        let sseqid = record.get(1).unwrap_or("").to_string();
        let slen = record.get(13).unwrap_or("0").parse::<usize>().unwrap_or(0);
        let sstart = record.get(8).unwrap_or("0").parse::<usize>().unwrap_or(0);
        let send = record.get(9).unwrap_or("0").parse::<usize>().unwrap_or(0);
        let bitscore = record.get(11).unwrap_or("0.0").parse::<f32>().unwrap_or(0.0);
        // Create a Record object and push it to the vector
        alignments.push(AlnRec { qseqid, sseqid, slen, sstart, send, bitscore });
    }

    // Collect unique qseqids and sseqids
    let mut qseqid_set: HashSet<String> = HashSet::new();
    let mut sseqid_set: HashSet<String> = HashSet::new();
    for record in &alignments {
        qseqid_set.insert(record.qseqid.clone());
        sseqid_set.insert(record.sseqid.clone());
    }
    let mut queries_vec: Vec<_> = qseqid_set.clone().into_iter().collect();
    queries_vec.sort();
    let mut subjects_vec: Vec<_> = sseqid_set.clone().into_iter().collect();
    subjects_vec.sort();
    // Create maps to store indices for qseqids and sseqids
    let qseqid_indices: HashMap<String, usize> = queries_vec.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();
    let sseqid_indices: HashMap<String, usize> = subjects_vec.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

    // Create triplet matrices for bitscore, sstart, and send
    let mut bitscore_triplet = TriMat::with_capacity((sseqid_set.len(), qseqid_set.len()), alignments.len());
    let mut sstart_triplet = TriMat::with_capacity((sseqid_set.len(), qseqid_set.len()), alignments.len());
    let mut send_triplet = TriMat::with_capacity((sseqid_set.len(), qseqid_set.len()), alignments.len());
    let mut aligned_triplet = TriMat::with_capacity((sseqid_set.len(), qseqid_set.len()), alignments.len());

    // Fill the triplet matrices with values
    for record in &alignments {
        let sseqid_index = *sseqid_indices.get(&record.sseqid).unwrap();
        let qseqid_index = *qseqid_indices.get(&record.qseqid).unwrap();

        bitscore_triplet.add_triplet(sseqid_index, qseqid_index, record.bitscore);
        sstart_triplet.add_triplet(sseqid_index, qseqid_index, record.sstart);
        send_triplet.add_triplet(sseqid_index, qseqid_index, record.send);
        aligned_triplet.add_triplet(sseqid_index, qseqid_index, 1);
    }


    // Convert triplet matrices into compressed sparse row matrices
    let bitscore_matrix = bitscore_triplet.to_csr();
    let sstart_matrix = sstart_triplet.to_csr();
    let send_matrix = send_triplet.to_csr();
    let aligned_matrix = aligned_triplet.to_csr();

    // Finally the subject to subject_len mapping:
    let sseqid_slen_map: HashMap<_, _> = alignments.into_iter()
        .map(|record| (record.sseqid, record.slen))
        .collect();    


    Ok((
        queries_vec,
        subjects_vec,
        aligned_matrix,
        bitscore_matrix, 
        sstart_matrix, 
        send_matrix,
        sseqid_slen_map,
        qseqid_indices, 
        sseqid_indices
    )) 
}

fn build_subject_cover(
    slen: usize,
    sstarts: Vec<usize>,
    sends: Vec<usize>,
) ->     Result<Array1<u32>, Box<dyn Error>> {

    println!("{:?}", slen);
    let mut cov_arr = Array1::<u32>::zeros(slen);

    for i in 0..sstarts.len() {
        cov_arr.slice_mut(s![sstarts[i]-1..sends[i]]).mapv_inplace(|x| x + 1);
    }

    Ok(cov_arr)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let opts = Args::parse();
    
    let (queries_vec, subjects_vec, aligned_matrix, bitscore_matrix, sstart_matrix, send_matrix, sseqid_slen_map, qseqid_indices, sseqid_indices) = read_alignment(&opts.aln)?;
    println!("Number of subjects: {}", subjects_vec.len());
    println!("Number of queries: {}", queries_vec.len());
    println!("Number of subject lengths: {}", sseqid_slen_map.len());
    println!("Aligned Matrix Shape: {:?}", aligned_matrix.shape());
    println!("Bitscore Matrix Shape: {:?}", bitscore_matrix.shape());
    println!("Sstart Matrix Shape: {:?}", sstart_matrix.shape());
    println!("Send Matrix Shape: {:?}", send_matrix.shape());



    // First coverage filter

    let strim_5 = 18;
    let strim_3 = 18;
    let sd_mean_cutoff = 4.0;

    for subj_i in 0..3 {
        let row_subj = &subjects_vec[subj_i];
        let row_slen = sseqid_slen_map[row_subj];
        let row_sstarts = sstart_matrix.slice_outer(subj_i..subj_i+1).data().to_vec();
        let row_sends = send_matrix.slice_outer(subj_i..subj_i+1).data().to_vec();

        let cov_arr = build_subject_cover(
            row_slen,
            row_sstarts,
            row_sends
        )?;
        println!("{:?}", cov_arr);
        
        let  cov_arr_f32: Array1<f32>;



        if cov_arr.len() > (strim_5 + strim_3) {
                cov_arr_f32 = cov_arr.slice(s![strim_5..(cov_arr.len() - strim_3)]).mapv(|x| x as f32);
        }  else {
            cov_arr_f32 = cov_arr.mapv(|x| x as f32);
        }
        let cov_mean = cov_arr_f32.mean_axis(Axis(0)).unwrap().into_scalar();
        let cov_std = cov_arr_f32.std_axis(Axis(0), 0.0).into_scalar();
        let cov_pass = (cov_std / cov_mean) < sd_mean_cutoff;
        println!("{:?}", cov_mean);
        println!("{:?}", cov_std);
        println!("{:?}", cov_pass);

        
    }

    Ok(())
}
