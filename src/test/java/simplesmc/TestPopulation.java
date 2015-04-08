package simplesmc;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.lang3.time.StopWatch;
import org.junit.Test;

import bayonet.distributions.NegativeBinomial;
import bayonet.distributions.Poisson;
import bayonet.smc.ParticlePopulation;
import bayonet.smc.ResamplingScheme;



public class TestPopulation
{
  @Test
  public void testResampling()
  {
    Random random = new Random(1);
    for (int nParticles = 10_000; nParticles <= 1_000_000; nParticles += 10_000)
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
  
  static int r = 2;
  static double p = 0.5;
  
  static double exactMean() 
  {
    return p * r / (1.0 - p);
  }
  
  static <T extends Number> double approximateMean(ParticlePopulation<T> population)
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
  
  static ParticlePopulation<Integer> negativeBinomialPopulation(Random random, int numberOfParticles)
  {
    final double proposalMean = 2.0;
    
    ArrayList<Integer> particles = new ArrayList<>();
    double [] logWeights = new double[numberOfParticles];

    // note: java is zero index
    for (int particleIndex = 0; particleIndex < numberOfParticles; particleIndex++)
    {
      int sample = Poisson.generate(random, proposalMean);
      
      particles.add(sample);
      logWeights[particleIndex] = NegativeBinomial.logDensity(sample, r, p) - Poisson.logDensity(sample, proposalMean);
    }
    
    return ParticlePopulation.buildDestructivelyFromLogWeights(logWeights, particles, 0.0);
  }
}
