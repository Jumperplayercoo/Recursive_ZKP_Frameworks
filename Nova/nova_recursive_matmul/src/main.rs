// Recursive matrix multiplication SNARK protocol in Nova

use anyhow::Result;
use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{circuit::StepCircuit, snark::RelaxedR1CSSNARKTrait, Engine},
};
use std::time::Instant;

#[path = "../utils/mod.rs"]
mod utils;
use utils::metrics::{format_size, measure_memory_usage};

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>;
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;

type F = <E1 as Engine>::Scalar;

const MATRIX_SIZE: usize = 1;
const PROOFS_PER_BATCH: usize = 2;
const NUM_BATCHES: usize = 2;

#[derive(Clone)]
struct MatmulCircuit {
    x: Vec<Vec<F>>,
    w: Vec<Vec<F>>,
    y: Vec<Vec<F>>,
}

impl StepCircuit<F> for MatmulCircuit {
    //Building Base Circuit
    fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        _inputs: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        for i in 0..MATRIX_SIZE {
            for j in 0..MATRIX_SIZE {
                let mut acc = AllocatedNum::alloc(cs.namespace(|| format!("acc_{}_{}", i, j)), || Ok(F::ZERO))?;
                for k in 0..MATRIX_SIZE {
                    let x_val = self.x[i][k];
                    let w_val = self.w[k][j];
                    let x_alloc = AllocatedNum::alloc(cs.namespace(|| format!("x_{}_{}", i, k)), || Ok(x_val))?;
                    let w_alloc = AllocatedNum::alloc(cs.namespace(|| format!("w_{}_{}", k, j)), || Ok(w_val))?;
                    let prod = x_alloc.mul(cs.namespace(|| format!("mul_{}_{}_{}", i, k, j)), &w_alloc)?;
                    acc = acc.add(cs.namespace(|| format!("add_{}_{}_{}", i, k, j)), &prod)?;
                }
                let y_val = self.y[i][j];
                let y_alloc = AllocatedNum::alloc(cs.namespace(|| format!("y_{}_{}", i, j)), || Ok(y_val))?;
                cs.enforce(
                    || format!("enforce_{}_{}", i, j),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + y_alloc.get_variable(),
                );
            }
        }
        Ok(vec![AllocatedNum::alloc(cs.namespace(|| "dummy_output"), || Ok(F::ZERO))?])
    }

    fn arity(&self) -> usize {
        1
    }
}

fn main() -> Result<()> {
    println!("=========================================================");
    println!("Nova Recursive Matrix Multiplication Circuit");
    println!("=========================================================");

    let ((), peak_memory) = measure_memory_usage(|| {
        let mut total_prove_time = std::time::Duration::ZERO;
        let mut total_circuit_build_time = std::time::Duration::ZERO;
        let mut total_recursive_verification_time = std::time::Duration::ZERO;
        let mut total_compressed_verification_time = std::time::Duration::ZERO;
        let mut total_compressed_size = 0;

        for batch in 0..NUM_BATCHES {
            println!("Batch {}/{}", batch + 1, NUM_BATCHES);
            //Zero matrices
            let mut circuits = vec![];
            for _ in 0..PROOFS_PER_BATCH {
                let x: Vec<Vec<F>> = (0..MATRIX_SIZE)
                    .map(|_| (0..MATRIX_SIZE).map(|_| F::ZERO).collect())
                    .collect();
                let w: Vec<Vec<F>> = (0..MATRIX_SIZE)
                    .map(|_| (0..MATRIX_SIZE).map(|_| F::ZERO).collect())
                    .collect();

                let mut y = vec![vec![F::ZERO; MATRIX_SIZE]; MATRIX_SIZE];
                for i in 0..MATRIX_SIZE {
                    for j in 0..MATRIX_SIZE {
                        for k in 0..MATRIX_SIZE {
                            y[i][j] += x[i][k] * w[k][j];
                        }
                    }
                }

                circuits.push(MatmulCircuit { x, w, y });
            }

            println!("\nGenerating Public Parameters...");
            let setup_start = Instant::now();
            let pp = PublicParams::<E1, E2, MatmulCircuit>::setup(
                &circuits[0],
                &*S1::ck_floor(),
                &*S2::ck_floor(),
            ).unwrap();
            let setup_time = setup_start.elapsed();
            println!("Public parameter setup time: {} us", setup_time.as_micros());
            total_circuit_build_time+=setup_time;

            println!(
                "Number of constraints per step (primary circuit): {}",
                pp.num_constraints().0
            );
            println!(
                "Number of constraints per step (secondary circuit): {}",
                pp.num_constraints().1
            );
            


            println!("\nGenerating RecursiveSNARK...");
            let mut recursive_snark = RecursiveSNARK::<E1, E2, MatmulCircuit>::new(
                &pp,
                &circuits[0],
                &[F::ZERO],
            ).unwrap();

            let mut batch_prove_time = std::time::Duration::ZERO;
            for (i, circuit) in circuits.iter().enumerate() {
                let step_start = Instant::now();
                let res = recursive_snark.prove_step(&pp, circuit);
                let step_time = step_start.elapsed();
                println!("Proof step {} took: {} us", i+1, step_time.as_micros());
                batch_prove_time+=step_time;
                assert!(res.is_ok());
            }
            println!("Total batch proof time: {} us", batch_prove_time.as_micros());
            total_prove_time += batch_prove_time;

            println!("Verifying RecursiveSNARK...");
            let verify_start = Instant::now();
            let res = recursive_snark.verify(&pp, PROOFS_PER_BATCH, &[F::ZERO]);
            let verify_time = verify_start.elapsed();
            println!("Verification time: {} us", verify_time.as_micros());
            total_recursive_verification_time += verify_time;
            assert!(res.is_ok());


            println!("\nGenerating a CompressedSNARK using Spartan with HyperKZG...");
            let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp).unwrap();
            let compress_start = Instant::now();
            let compressed_snark = CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark).unwrap();
            println!("Compressed SNARK proof time: {} us", compress_start.elapsed().as_micros());

            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
            bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
            let compressed_bytes = encoder.finish().unwrap();
            let size = compressed_bytes.len();
            total_compressed_size += size;
            println!("Compressed SNARK size: {}", format_size(size as u64));

            println!("Verifying CompressedSNARK...");
            let compressed_verify_start = Instant::now();
            let res = compressed_snark.verify(&vk, PROOFS_PER_BATCH, &[F::ZERO]);
            let compressed_verify_time = compressed_verify_start.elapsed();
            println!("Compressed SNARK verification time: {:?} us", compressed_verify_time.as_micros());
            total_compressed_verification_time += compressed_verify_time;
            assert!(res.is_ok());

            println!("=========================================================");
        }

        println!("\n=============== Summary ===============");
        println!("Total Circuit Build Time: {} us", total_circuit_build_time.as_micros());
        println!("Total Proof Generation Time: {} us", total_prove_time.as_micros());
        println!("Final Recursive Verification Time: {} us", total_recursive_verification_time.as_micros());
        println!("Final Compressed Verification Time: {} us", total_compressed_verification_time.as_micros());
        println!("Total Compressed Proof Size:{}", format_size(total_compressed_size as u64));
    });

    println!("Peak memory usage: {}", format_size(peak_memory));
    println!("=========================================================");

    Ok(())
}
