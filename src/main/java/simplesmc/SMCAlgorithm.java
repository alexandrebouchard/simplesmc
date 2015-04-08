package simplesmc;

import java.util.Arrays;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

import org.apache.commons.lang3.tuple.Pair;

import bayonet.smc.ParticlePopulation;



public class SMCAlgorithm<P>
{
  public final SMCProposal<P> proposal;
  private final SMCOptions options;
  private final Random[] randoms;
  
  public ParticlePopulation<P> sample()
  {
    ParticlePopulation<P> currentPopulation = propose(null, 0);
    
    int nSMCIterations = proposal.nIterations();
    
    for (int previousIteration = 0; previousIteration < nSMCIterations - 1; previousIteration++)
    {
      currentPopulation = propose(currentPopulation, previousIteration);
      if (currentPopulation.getRelativeESS() < options.essThreshold &&
          previousIteration < nSMCIterations - 2)
        currentPopulation = currentPopulation.resample(options.random, options.resamplingScheme);
    }
    
    return currentPopulation;
  }
  
  private ParticlePopulation<P> propose(ParticlePopulation<P> currentPopulation, int previousIteration)
  {
    boolean isInitial = currentPopulation == null;
    
    final double [] logWeights = new double[options.nParticles];
    @SuppressWarnings("unchecked")
    final P [] particles = (P[]) new Object[options.nParticles];
    
    IntStream.range(0, options.nParticles).parallel().forEach((particleIndex) -> 
    {
      Pair<Double, P> proposed = isInitial ?
        proposal.proposeInitial(randoms[particleIndex]) :
        proposal.proposeNext(previousIteration, randoms[particleIndex], currentPopulation.particles.get(particleIndex));
      logWeights[particleIndex] = 
        proposed.getLeft().doubleValue() + 
        (isInitial ? 0.0 : Math.log(currentPopulation.getNormalizedWeight(particleIndex)));
        particles[particleIndex] = (proposed.getRight());
    });
    
    return ParticlePopulation.buildDestructivelyFromLogWeights(
        logWeights, 
        Arrays.asList(particles),
        isInitial ? 0.0 : currentPopulation.logScaling);
  }

  public SMCAlgorithm(SMCProposal<P> proposal, SMCOptions options)
  {
    this.proposal = proposal;
    this.options = options;
    this.randoms = new Random[options.nParticles];
    SplittableRandom splitRandom = new SplittableRandom(options.random.nextLong());
    for (int i = 0; i < options.nParticles; i++)
      this.randoms[i] = new Random(splitRandom.split().nextLong());
  }
}
