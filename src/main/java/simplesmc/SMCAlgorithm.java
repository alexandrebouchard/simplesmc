package simplesmc;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.commons.lang3.tuple.Pair;

import bayonet.smc.ParticlePopulation;
import bayonet.smc.ResamplingScheme;
import briefj.opt.Option;



public class SMCAlgorithm<P>
{
  final SMCProposal<P> proposal;
  final SMCOptions options;
  
  public static class SMCOptions
  {
    @Option
    public double essThreshold = 0.5;

    @Option
    public Random random = new Random(1);
    
    @Option
    public int nParticles = 1000000;
    
    @Option
    public ResamplingScheme resamplingScheme = ResamplingScheme.MULTINOMIAL;
  }
  
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
    
    double [] logWeights = new double[options.nParticles];
    ArrayList<P> particles = new ArrayList<>();
    StopWatch watch = new StopWatch();
    watch.start();
    for (int particleIndex = 0; particleIndex < options.nParticles; particleIndex++)
    {
      Pair<Double, P> proposed = isInitial ?
          proposal.proposeInitial(options.random) :
          proposal.proposeNext(previousIteration, options.random, currentPopulation.particles.get(particleIndex));
      logWeights[particleIndex] = 
          proposed.getLeft().doubleValue() + 
          (isInitial ? 0.0 : Math.log(currentPopulation.getNormalizedWeight(particleIndex)));
      particles.add(proposed.getRight());
    }
    System.out.println(watch.getTime());
    
    return ParticlePopulation.buildDestructivelyFromLogWeights(
        logWeights, 
        particles, 
        isInitial ? 0.0 : currentPopulation.logScaling);
  }

  public SMCAlgorithm(SMCProposal<P> proposal, SMCOptions options)
  {
    this.proposal = proposal;
    this.options = options;
  }
}
