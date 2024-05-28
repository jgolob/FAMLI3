extern crate clap;
extern crate ndarray;
extern crate rayon;
extern crate serde_json;
extern crate serde;
extern crate num_cpus;
extern crate log;
extern crate env_logger;
extern crate chrono;

use clap::Parser;
use std::fs::File;
use std::io::Write;
use serde_json::to_string_pretty;
use serde::Serialize;
use std::io::{BufReader, Read};
use std::error::Error;
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, s, Axis};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

use log::{info, error, LevelFilter};
use env_logger::Builder;
use std::time::Instant;


#[derive(Parser, Debug)] 
#[command(version, about)]
struct Args {
    #[arg(short, long, value_name = "FILE", required = true)]
    aln: String,
    #[arg(short, long, value_name = "FILE", required = true)]
    output: String,
    #[arg(long, default_value_t=18)]
    strim_5: usize,
    #[arg(long, default_value_t=18)]
    strim_3: usize,
    #[arg(long, default_value_t=3.0)]
    sd_mean_cutoff: f32,
    #[arg(long, default_value_t=0.9)]
    filter_fract: f64,
    #[arg(long, default_value_t=1000)]
    max_iterations: usize,
    #[arg(long, default_value_t = num_cpus::get())]
    max_cpus: usize,    
    
}

struct Alignments {
    qseqid_set: HashSet<String>,
    sseqid_set: HashSet<String>,
    slen_map: HashMap<String, usize>, // sseqid: slen
    sstarts_map: HashMap<String, HashMap<String, usize>>, // sseqid / qseqid / sstart
    sends_map: HashMap<String, HashMap<String, usize>>, // sseqid / qseqid // send
    q_s_bitscore_map: HashMap<String, HashMap<String, f32>>, // qseqid / sseqid / bitscore
}

#[derive(Clone, Serialize)]
struct SubjectCoverage {
    cov: f32, // Fraction of subject covered with at least one read
    depth: f32, // Average number of reads per location
    std: f32, // standard deviation of the depth
    nreads: usize, // number of reads assigned to this subject
    length: usize,  // How long is this subject
    id: String, // subject ID
}


fn read_alignment(file_path: &str) -> 
    Result<Alignments, Box<dyn Error>> {
    info!("Attempting to open alignment and autodetect if gzipped or not");

    let start = Instant::now(); // Record start time

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

    
    info!("Starting to read in alignments");
    
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

    let end = Instant::now(); // Record end time
    let elapsed = end.duration_since(start).as_secs();

    info!("Completed reading of alignments in {:?} seconds", elapsed);

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
            error!("Send {} is greater than slen {}!", sends[i], slen);
        }
        cov_arr.slice_mut(s![sstarts[i]-1..sends[i]]).mapv_inplace(|x| x + 1);
    }

    cov_arr
}

fn coverage_filter(
    alignments: &Alignments,
    strim_5: usize,
    strim_3: usize,
    sd_mean_cutoff: f32,
) -> (HashSet<String>, HashMap<String, SubjectCoverage>) {

    // Thread safe place to add our filtered subjects
    let subjects_to_be_removed: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    let subject_coverages:  Arc<Mutex<HashMap<String, SubjectCoverage>>> = Arc::new(Mutex::new(HashMap::new()));

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
        let cov_cov = cov_arr_f32.iter().filter(|&val| *val > 0.0).count() as f32 / cov_arr_f32.len() as f32;
        let cov_pass = (cov_std / cov_mean) < sd_mean_cutoff;
        if !cov_pass {
            // Acquire lock for writing
            let mut subjects_to_be_removed_set = subjects_to_be_removed.lock().unwrap();
            subjects_to_be_removed_set.insert(subj.clone());
        }
        {
            let mut map = subject_coverages.lock().unwrap();
            map.insert(subj.clone(), SubjectCoverage {
                id: subj.clone(),
                depth: cov_mean,
                std: cov_std,
                length: cov_arr.len(),
                nreads: s_queries.len(),
                cov: cov_cov,
            });
        }
    }); // End first coverage pass iteration
    // Acquire lock for reading
    let subjects_to_be_removed_set = subjects_to_be_removed.lock().unwrap();

    // Clone the HashMap within the Arc<Mutex<HashMap>>
    let subject_coverages_map: HashMap<String, SubjectCoverage> = {
        let map = subject_coverages.lock().unwrap();
        map.clone()
    };


    (
        alignments.sseqid_set.difference(
            &subjects_to_be_removed_set
        ).cloned().collect(),
        subject_coverages_map
    )

}

fn bitscore_filter(
    alignments: &Alignments,
    max_iterations: usize,
    filter_frac: f64,
) -> Alignments {

    // Get our current queries as an ordered vector.
    let mut queries_vec: Vec<String> = alignments.qseqid_set.clone()
        .into_iter()
        .collect::<Vec<String>>();
    queries_vec.sort();

    // Build our 'alignment' scores as a vector of HashMaps<sseqid, alignment_core>, with one row per query.
    // The initial value is just the bitscores normalized into a 'relative bitscore per query'
    // Sums up to 1.0 for each query.
    let mut aln_scores = queries_vec.clone().into_par_iter()
        .map(|query| {
            let qbitscores = alignments.q_s_bitscore_map[&query.clone()].clone();
            let qbitscore_total: f64 = qbitscores.values().sum::<f32>() as f64;

            qbitscores.iter()
                .map(|(subj, bs)| (subj.clone(), *bs as f64 / qbitscore_total ))
                .collect::<HashMap<String, f64>>()
        })
        .collect::<Vec<HashMap<String, f64>>>();
    
    for iter in 0..max_iterations {
        // Calculate a per-subject weighting. 
        // The total alignment weights assigned to this subject divided by the weight.
        let mut subject_weights: HashMap<String, f64> = HashMap::new();
        aln_scores.iter().for_each(|qscores_map|{
            for (subj, score) in qscores_map.iter() {
                *subject_weights.entry(subj.clone()).or_insert(0.0) += *score;
            }
        });

        // Thread safe place to store if any alignments were filtered
        let iter_n_filtered = Arc::new(Mutex::new(0));

        // Re-calculate alignment scores with these subject weights and filter
        aln_scores = aln_scores.clone().into_par_iter()
        .map(|q_s_scores| {
            // Multiply each subject score by the current subject weight
            let mut adjusted_qs_scores = q_s_scores.iter()
                .map(|(subj, score)|{
                    (subj.clone(), score * subject_weights[subj])
                }).collect::<HashMap<String, f64>>();
            // Get the new total sum of scores for this query
            let q_total_score: f64 = adjusted_qs_scores.values().sum();
            // Then renormalize so the total scores equal 1 for this query by dividing by total
            for score in adjusted_qs_scores.values_mut() {
                *score = *score / q_total_score;
            }
            let filtered_adjusted_qs_scores: HashMap<String, f64> = 
            if adjusted_qs_scores.len() == 1 {
                adjusted_qs_scores
            } else {
                // Filtering.
                // Identify the max score for this query:
                let q_max_score = adjusted_qs_scores.values()
                    .fold(f64::NEG_INFINITY, |max, value| f64::max(max, *value));

                let q_cutoff = q_max_score * filter_frac;
                
                let filtered_adjusted_qs_scores: HashMap<String, f64> = adjusted_qs_scores.iter()
                    .filter(|&(_, &value)| value >= q_cutoff)
                    .map(|(key, &value)| (key.clone(), value))
                    .collect();
                if adjusted_qs_scores.len() != filtered_adjusted_qs_scores.len() {
                    let mut iter_n_filtered_lock = iter_n_filtered.lock().unwrap();
                    *iter_n_filtered_lock += adjusted_qs_scores.len() - filtered_adjusted_qs_scores.len();
                }
                filtered_adjusted_qs_scores    
            };
            filtered_adjusted_qs_scores
        })
        .collect::<Vec<HashMap<String, f64>>>();

        let iter_filtered_final  = *iter_n_filtered.lock().unwrap();

        info!("Filtered {:?} alignments on iteration {:?}", iter_filtered_final, iter+1);

        if iter_filtered_final == 0 {
            break;
        }
    } // End iteration
    info!("Generating a new set of alignments that passed this filter.");
    // Create a new Alignments structure just of those alignments that passed the filter...
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

    for (q_i, qseqid )in queries_vec.iter().enumerate() {
        qseqid_set.insert(qseqid.clone());
        let q_aln_scores = &aln_scores[q_i];
        sseqid_set.extend(q_aln_scores.keys().cloned());
        for sseqid in q_aln_scores.keys() {
            slen_map.insert(sseqid.clone(), alignments.slen_map[sseqid].clone());
            sstarts_map.entry(
                sseqid.clone()).or_insert(HashMap::new()
            ).insert(qseqid.clone(), alignments.sstarts_map[sseqid][qseqid].clone());
            sends_map.entry(
                sseqid.clone()).or_insert(HashMap::new()
            ).insert(qseqid.clone(), alignments.sends_map[sseqid][qseqid].clone());
            q_s_bitscore_map.entry(
                qseqid.clone()).or_insert(HashMap::new()
            ).insert(sseqid.clone(), alignments.q_s_bitscore_map[qseqid][sseqid].clone());
        }
    }
    Alignments{
        qseqid_set,
        sseqid_set,
        slen_map,
        sstarts_map,
        sends_map,
        q_s_bitscore_map,
    }
}


fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let opts = Args::parse();
    
    // Set the global thread pool with the specified number of threads
    rayon::ThreadPoolBuilder::new().num_threads(opts.max_cpus).build_global().unwrap();

    // Initialize the logger with a custom format
    Builder::new()
    .format(|buf, record| {
        writeln!(
            buf,
            "{} {:<8} [FAMLI3] {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            record.level(),
            record.args()
        )
    })
    .filter(None, LevelFilter::Info)
    .init();


    let mut alignments = read_alignment(&opts.aln)?;

    info!("Number of Subjects: {:?}", alignments.sseqid_set.len());
    info!("Number of Queries: {:?}", alignments.qseqid_set.len());
    

    // 1.  First subject depth / evenness filter.
    info!("Starting First Coverage Filter");

    let start_cov_filt_1 = Instant::now(); // Record start time

    (alignments.sseqid_set, _) = coverage_filter(
        &alignments,
        opts.strim_5,
        opts.strim_3,
        opts.sd_mean_cutoff,
    );

    let end_cov_filt_1 = Instant::now(); // Record end time
    let elapsed_cov_filt_1 = end_cov_filt_1.duration_since(start_cov_filt_1).as_secs();
    info!("Completed first coverage filter in {:?} seconds", elapsed_cov_filt_1);    
    info!("There are {:?} subjects remaining after the first coverage filter.", alignments.sseqid_set.len());


    // BITSCORE FILTER
    info!("Starting bitscore filter.");

    let post_bs_filter_aln = bitscore_filter(
        &alignments,
        opts.max_iterations,
        opts.filter_fract,
    );

    // Final Coverage filter
    //let mut final_subject_coverages: HashMap<String, SubjectCoverage> = HashMap::new();
    let (_, final_subject_coverages) = coverage_filter(
        &post_bs_filter_aln,
        opts.strim_5,
        opts.strim_3,
        opts.sd_mean_cutoff,
    );

    
    let subj_cov_vec: Vec<_> = final_subject_coverages.values().map(|subj_cov|{
        subj_cov.clone()
    })
    .collect();
    
    let json_data = to_string_pretty(&subj_cov_vec)?;

    let mut file = File::create(&opts.output)?;
    file.write_all(json_data.as_bytes())?;
    
    Ok(())
}
