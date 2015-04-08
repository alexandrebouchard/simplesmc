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
 * 
 * Terminology: a population is a list of particles, their weights, and a normalization constant estimate
 *
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

    // note: java is zero index
    for (int particleIndex = 0; particleIndex < numberOfParticles; particleIndex++)
    {
      /* 
       * Construct a ParticlePopulation using importance sampling
       * where the proposal is Poisson(proposalMean) and the target distribution is
       * NegativeBinomial(r, p)
       * 
       * Note that r and p are defined below as global variables.
       * 
       * Hints:
       * 
       * - you need to populate the particles and their weights
       * - to call a static (standard function) use for example Poisson.generate(random, proposalMean)
       *   where Poisson is the name of the file. 
       * - dot has a different meaning than in R. It is a bit like $
       * - after the dot use control-space for the auto-complete!
       * - you can also right click and select "Open Declaration" when you want to see the source of 
       *   a function you are calling
       * - you also need to import classes (files) you reference to, but use the auto-complete for that!
       * 
       * To test your code:
       * 
       * - right click on the file TestPopulation
       * - select "Run as" > "JUnit test"
       * - this will run the test called testIS() below
       * - you should see a green bar
       */
      /* startRem throw new RuntimeException(); */
      int sample = Poisson.generate(random, proposalMean);
      
      particles.add(sample);
      logWeights[particleIndex] = NegativeBinomial.logDensity(sample, r, p) - Poisson.logDensity(sample, proposalMean);
      /* endRem */
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
  
  /*
   * Now create a new function, called naiveResample, which takes as input a population,
   * and output a new population obtained after multinomial re-sampling. 
   * 
   * Write a new test called testResampling() similar to the one above, that test your code
   * with 10k, 20k, 30k, .., 100k particles.
   * 
   * Comment on the empirical running time as a function of the number of particles.
   * Can we do better?
   * 
   * Hints:
   * 
   * - you can use ParticlePopulation.buildEquallyWeighted(..) to save yourself the trouble of
   *   creating a vector of weights all equal to 1/N
   * - you can also use population.sample(..)
   */
  
  /* startRem */
  
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

  static <T> ParticlePopulation<T> naiveResample(Random random, ParticlePopulation<T> population)
  {
    ArrayList<T> particles = new ArrayList<>();
    
    
    for (int particleIndex = 0; particleIndex < population.nParticles(); particleIndex++)
    {
      particles.add(population.sample(random));
    }
    
    return ParticlePopulation.buildEquallyWeighted(particles, population.logScaling);
  }

  /* endRem */
}
