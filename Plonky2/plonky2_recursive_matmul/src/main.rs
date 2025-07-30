// Recursive matrix multiplication SNARK protocol in Plonky2

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::{CircuitConfig, CircuitData};
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::plonk::proof::ProofWithPublicInputs;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[path = "../utils/mod.rs"]
mod utils;
use utils::metrics::{format_size, measure_memory_usage};


const MATRIX_SIZE: usize = 1;
const PROOFS_PER_BATCH: usize = 1;
const NUM_BATCHES: usize = 5;


const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;

static MEMORY_USAGE: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn memory_usage(memory: u64) {
    if let Ok(mut usage) = MEMORY_USAGE.lock() {
        usage.push(memory);
    }
}

fn get_peak_memory() -> u64 {
    MEMORY_USAGE
        .lock()
        .map(|usage| usage.iter().max().copied().unwrap_or(0))
        .unwrap_or(0)
}

fn build_matmul_circuit(
    x: &[Vec<F>],
    w: &[Vec<F>],
    y: &[Vec<F>],
    builder: &mut CircuitBuilder<F, D>,
) -> PartialWitness<F> {
    let mut x_targets = vec![vec![]; MATRIX_SIZE];
    let mut y_targets = vec![vec![]; MATRIX_SIZE];
    let mut w_targets = vec![vec![]; MATRIX_SIZE];

    for i in 0..MATRIX_SIZE {
        for _j in 0..MATRIX_SIZE {
            let x_t = builder.add_virtual_public_input();
            let y_t = builder.add_virtual_public_input();
            x_targets[i].push(x_t);
            y_targets[i].push(y_t);
        }
    }

    for i in 0..MATRIX_SIZE {
        for _j in 0..MATRIX_SIZE {
            let w_t = builder.add_virtual_target();
            w_targets[i].push(w_t);
        }
    }

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            let mut acc = builder.zero();
            for k in 0..MATRIX_SIZE {
                let mul = builder.mul(x_targets[i][k], w_targets[k][j]);
                acc = builder.add(acc, mul);
            }
            builder.connect(acc, y_targets[i][j]);
        }
    }

    let mut pw = PartialWitness::new();

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            pw.set_target(x_targets[i][j], x[i][j]);
            pw.set_target(y_targets[i][j], y[i][j]);
            pw.set_target(w_targets[i][j], w[i][j]);
        }
    }

    pw
}

fn build_base_matmul_circuit(
    x: &[Vec<F>],
    w: &[Vec<F>],
    y: &[Vec<F>],
) -> Result<(CircuitData<F, C, D>, ProofWithPublicInputs<F, C, D>, Duration, Duration)> {
    println!("Building base matrix multiplication circuit...");

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let pw = build_matmul_circuit(x, w, y, &mut builder);

    println!("Base circuit gates: {}", builder.num_gates());

    let build_start = Instant::now();
    let data = builder.build::<C>();
    let build_time = build_start.elapsed();
    println!("Base circuit build/setup time: {} us", build_time.as_micros());

    println!("Generating base proof...");
    let proof_start = Instant::now();
    let (proof_result, memory_used) = measure_memory_usage(|| data.prove(pw));
    let proof_time = proof_start.elapsed();
    memory_usage(memory_used);
    let proof = proof_result?;
    println!("Base proof generation time: {} us", proof_time.as_micros());
    println!("Base proof size: {} us", format_size(proof.to_bytes().len() as u64));
    println!("Base memory used: {}", format_size(memory_used));
    println!("=========================================================");
    

    Ok((data, proof, build_time, proof_time))
}


fn build_recursive_matmul(
    prev_data: &CircuitData<F, C, D>,
    prev_proof: &ProofWithPublicInputs<F, C, D>,
    x: &[Vec<F>],
    w: &[Vec<F>],
    y: &[Vec<F>],
) -> Result<(CircuitData<F, C, D>, ProofWithPublicInputs<F, C, D>, Duration, Duration)> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let cap_height = prev_data.common.config.fri_config.cap_height;
    let prev_verifier_data = builder.add_virtual_verifier_data(cap_height);
    let prev_proof_t = builder.add_virtual_proof_with_pis(&prev_data.common);

    let mut pw = PartialWitness::new();
    pw.set_proof_with_pis_target(&prev_proof_t, prev_proof);
    pw.set_verifier_data_target(&prev_verifier_data, &prev_data.verifier_only);

    builder.verify_proof::<C>(&prev_proof_t, &prev_verifier_data, &prev_data.common);

    let mut matmul_pw = build_matmul_circuit(x, w, y, &mut builder);
    for (target, value) in matmul_pw.target_values.drain() {
        pw.set_target(target, value);
    }

    let build_start = Instant::now();
    let data = builder.build::<C>();
    let build_time = build_start.elapsed();

    let proof_start = Instant::now();
    let (proof_result, memory_used) = measure_memory_usage(|| data.prove(pw));
    let proof_time = proof_start.elapsed();
    memory_usage(memory_used);

    Ok((data, proof_result?, build_time, proof_time))
}

fn main() -> Result<()> {
    println!("=========================================================");
    println!("Plonky2 Recursive Matrix Multiplication Circuit");
    println!("=========================================================");
    let mut total_proof_time = Duration::ZERO;
    let mut total_build_time = Duration::ZERO;
    let mut total_verify_time = Duration::ZERO;
    let mut total_proof_size = 0;

    let x = vec![vec![F::ZERO; MATRIX_SIZE]; MATRIX_SIZE];
    let w = vec![vec![F::ZERO; MATRIX_SIZE]; MATRIX_SIZE]; 
    let mut y = vec![vec![F::ZERO; MATRIX_SIZE]; MATRIX_SIZE];

    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            for k in 0..MATRIX_SIZE {
                y[i][j] += x[i][k] * w[k][j];
            }
        }
    }

    let (mut current_data, mut current_proof, base_build_time, base_proof_time) =
        build_base_matmul_circuit(&x, &w, &y)?;
    total_build_time += base_build_time;
    total_proof_time += base_proof_time;
    total_proof_size += current_proof.to_bytes().len();

    for batch in 0..NUM_BATCHES {
        println!("Generating batch {}/{} ...", batch + 1, NUM_BATCHES);
        let mut batch_build_time = Duration::ZERO;
        let mut batch_proof_time = Duration::ZERO;
        let mut batch_proof_size = 0;

        for proof_index in 0..PROOFS_PER_BATCH {
            let proof_start = Instant::now();
            let (new_data, new_proof, build_time, proof_time) =
                build_recursive_matmul(&current_data, &current_proof, &x, &w, &y)?;
            let proof_time = proof_start.elapsed();

            println!(
                "  Proof step {} took: {} us",
                proof_index + 1,
                proof_time.as_micros()
            );

            batch_build_time += build_time;
            batch_proof_time += proof_time;
            batch_proof_size += new_proof.to_bytes().len();

            total_build_time += build_time;
            total_proof_time += proof_time;
            total_proof_size += new_proof.to_bytes().len();

            current_data = new_data;
            current_proof = new_proof;
        }

        println!("Batch {} build time: {} us", batch + 1, batch_build_time.as_micros());
        println!("Batch {} proof time: {} us", batch + 1, batch_proof_time.as_micros());
        println!("Batch {} proof size: {}", batch + 1, format_size(batch_proof_size as u64));
        println!("=========================================================");
    }

    let now = Instant::now();
    current_data.verify(current_proof.clone())?;
    total_verify_time += now.elapsed();
    println!("\n=============== Summary ===============");
    println!("Total Circuit Build Time: {} us", total_build_time.as_micros());
    println!("Total Proof Generation Time: {} us", total_proof_time.as_micros());
    println!("Final Recursive Verification Time: {} us", total_verify_time.as_micros());
    println!("Total Proof Size: {}", format_size(total_proof_size as u64));
    println!("Peak Memory Usage: {}", format_size(get_peak_memory()));
    println!("=========================================================");
    Ok(())
}
