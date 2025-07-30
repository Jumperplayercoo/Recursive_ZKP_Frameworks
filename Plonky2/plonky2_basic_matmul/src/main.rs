// Basic matrix multiplication SNARK protocol in Plonky2
// Adapted from Nojan Sheybani's https://github.com/ACESLabUCSD/ZeroKnowledgeFrameworksSurvey

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use std::time::{Duration, Instant};

#[path = "../utils/mod.rs"]
mod utils;
use utils::metrics::{format_size, measure_memory_usage};

const BENCHMARK_ROUNDS: u32 = 10;
const MATRIX_SIZE: usize = 5;

pub type F = <C as GenericConfig<D>>::F;
const D: usize = 2;
type C = PoseidonGoldilocksConfig;

fn build_matmul_circuit(
    x: &[Vec<F>],
    w: &[Vec<F>],
    y: &[Vec<F>],
    builder: &mut CircuitBuilder<F, D>,
) -> (PartialWitness<F>, Vec<Vec<plonky2::iop::target::Target>>, Vec<Vec<plonky2::iop::target::Target>>, Vec<Vec<plonky2::iop::target::Target>>) {
    let mut x_targets = vec![vec![]; MATRIX_SIZE];
    let mut y_targets = vec![vec![]; MATRIX_SIZE];
    let mut w_targets = vec![vec![]; MATRIX_SIZE];

    // X and Y are public inputs
    for i in 0..MATRIX_SIZE {
        for _j in 0..MATRIX_SIZE {
            let x_t = builder.add_virtual_public_input();
            let y_t = builder.add_virtual_public_input();
            x_targets[i].push(x_t);
            y_targets[i].push(y_t);
        }
    }

    // Add W as private witness
    for i in 0..MATRIX_SIZE {
        for _j in 0..MATRIX_SIZE {
            let w_t = builder.add_virtual_target();
            w_targets[i].push(w_t);
        }
    }

    // Enforce matrix multiplication X * W = Y
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

    // Assign values to witness
    for i in 0..MATRIX_SIZE {
        for j in 0..MATRIX_SIZE {
            pw.set_target(x_targets[i][j], x[i][j]);
            pw.set_target(y_targets[i][j], y[i][j]);
            pw.set_target(w_targets[i][j], w[i][j]);
        }
    }

    (pw, x_targets, y_targets, w_targets)
}

fn main() -> Result<()> {
    println!("=========================================================");
    println!("Plonky2 Basic Matrix Multiplication Circuit");
    println!("=========================================================");
    let mut total_setup_time = Duration::ZERO;
    let mut total_proof_time = Duration::ZERO;
    let mut total_verify_time = Duration::ZERO;
    let mut total_proof_size = 0;
    let mut total_memory_used = 0;
    let mut num_gates=0;
 
    let config = CircuitConfig::standard_recursion_config();

    for _round_number in 0..BENCHMARK_ROUNDS {
        // Zero matrices
        let x = vec![vec![F::ZERO; MATRIX_SIZE]; MATRIX_SIZE];
        let w = vec![vec![F::ZERO; MATRIX_SIZE]; MATRIX_SIZE]; 
        let mut y = vec![vec![F::ZERO; MATRIX_SIZE]; MATRIX_SIZE];

        // Compute Y = X * W outside the circuit
        for i in 0..MATRIX_SIZE {
            for j in 0..MATRIX_SIZE {
                for k in 0..MATRIX_SIZE {
                    y[i][j] += x[i][k] * w[k][j];
                }
            }
        }

        let mut builder = CircuitBuilder::<F, D>::new(config.clone());
        let (pw, _x_targets, _y_targets, _w_targets) = build_matmul_circuit(&x, &w, &y, &mut builder);
        
        //Number of gates in the circuit
        num_gates = builder.num_gates();

        //Circuit build time
        let t = Instant::now();
        let data = builder.build::<C>();
        total_setup_time += t.elapsed();
        
        //Proof generation time
        let now = Instant::now();
        let (proof_result, memory_used) = measure_memory_usage(|| data.prove(pw));
        let matmul_proof = proof_result?;
        let prove_time = now.elapsed();
        total_proof_time+=prove_time;

        //Proof size
        total_proof_size+=matmul_proof.to_bytes().len() as u64;

        //Proof verification time
        let now = Instant::now();
        data.verify(matmul_proof)?;
        let verify_time = now.elapsed();
        total_verify_time += verify_time;
        
        //Memory used for proof generation
        total_memory_used+=memory_used;
    }

    println!("Number of Gates: {}", num_gates);
    println!("Circuit Build Time: {} us", (total_setup_time / BENCHMARK_ROUNDS).as_micros());
    println!("Proof Generation Time: {} us", (total_proof_time / BENCHMARK_ROUNDS).as_micros());
    println!("Verification Time: {} us", (total_verify_time / BENCHMARK_ROUNDS).as_micros());
    println!("Proof Size: {}", format_size(total_proof_size/BENCHMARK_ROUNDS as u64));
    println!("Total Memory Used: {}", format_size(total_memory_used/BENCHMARK_ROUNDS as u64));
    Ok(())
    
}
