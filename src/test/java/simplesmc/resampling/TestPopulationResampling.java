package simplesmc.resampling;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.lang3.time.StopWatch;
import org.junit.Assert;
import org.junit.Test;

import bayonet.distributions.NegativeBinomial;
import bayonet.distributions.Poisson;
import bayonet.smc.ParticlePopulation;
import bayonet.smc.ResamplingScheme;


/**
 * Some simple test cases on the particle population datastructure, and resampling algorithms.
 * 
 * Terminology: a population is a list of particles, their weights, and a normalization constant estimate
 */
public class TestPopulationResampling
{
  /**
   * static: means that we are defining a standard function (not attached to an object)
   * ParticlePopulation<Integer> means that the function will return a population in which the particles in the are integers
   */
  static ParticlePopulation<Integer> negativeBinomialPopulation(Random random, int numberOfParticles)
  {
    final double proposalMean = 2.0;
    
    ArrayList<Integer> particles = new ArrayList<>();
    double [] logWeights = new double[numberOfParticles];

    for (int particleIndex = 0; particleIndex < numberOfParticles; particleIndex++)
    {
      int sample = Poisson.generate(random, proposalMean);
      particles.add(sample);
      logWeights[particleIndex] = NegativeBinomial.logDensity(sample, r, p) - Poisson.logDensity(sample, proposalMean);
    }
    
    // This exponentiates and normalizes the weights (destructively, meaning that the input array is
    // modified in place
    return ParticlePopulation.buildDestructivelyFromLogWeights(logWeights, particles, 0.0);
  }
  
  static int r = 2;
  static double p = 0.5;
  
  @Test
  public void testIS()
  {
    int nParticles = 1_000_000;
    Random random = new Random(1);
    ParticlePopulation<Integer> population = negativeBinomialPopulation(random, nParticles);
    
    // check that the mean of the population matches the analytic mean
    double exact  = exactMean();
    double approx = approximateMean(population);
    System.out.println("exact mean  = " + exact);
    System.out.println("approx mean = " + approx);
    Assert.assertTrue(Math.abs(exact - approx)/exact < 0.05);
  }
  
  static double exactMean() 
  {
    return p * r / (1.0 - p);
  }
  
  /*
   * This one is public to make it accessible from files that are not in the same package,
   * (directory), in case we need it later on
   */
  public static <T extends Number> double approximateMean(ParticlePopulation<T> population)
  {
    double sum = 0.0;
    
    for (int particleIndex = 0; particleIndex < population.nParticles(); particleIndex++)
    {
      Number particle = population.particles.get(particleIndex);
      double weight = population.getNormalizedWeight(particleIndex);
      sum += particle.doubleValue() * weight;
    }
    
    return sum;
  }
  
  @Test
  public void testResampling()
  {
    Random random = new Random(1);
    for (int nParticles = 10_000; nParticles <= 100_000; nParticles += 10_000)
    {
      StopWatch stopWatch = new StopWatch();
      stopWatch.start();
      ParticlePopulation<Integer> population = negativeBinomialPopulation(random, nParticles);
      check(population);
      ParticlePopulation<Integer> resampled = 
        //naiveResample(random, population);
        population.resample(random, ResamplingScheme.MULTINOMIAL);
      check(resampled);
      System.out.println("time=" + stopWatch.getTime() + "ms");
    }
  }
  
  static void check(ParticlePopulation<Integer> population)
  {
    double error = Math.abs(exactMean() - approximateMean(population));
    double ess = population.getRelativeESS();
    System.out.println("nParticles=" + population.nParticles() + ",error=" + error + ",ess=" + ess);
  }

//  static <T> ParticlePopulation<T> naiveResample(Random random, ParticlePopulation<T> population)
//  {
//    ArrayList<T> particles = new ArrayList<>();
//    for (int particleIndex = 0; particleIndex < population.nParticles(); particleIndex++)
//      particles.add(population.sample(random));
//    return ParticlePopulation.buildEquallyWeighted(particles, population.logScaling);
//  }
}
