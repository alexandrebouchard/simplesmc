package simplesmc;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;

import bayonet.distributions.ExhaustiveDebugRandom;
import bayonet.distributions.Random;
import simplesmc.hmm.HMMProblemSpecification;
import simplesmc.hmm.HMMUtils;
import simplesmc.hmm.ToyHMMParams;

public class TestUnbiased
{
  @Test
  public void hmm()
  {
    // Create a synthetic dataset
    Random random = new Random(1);
    ToyHMMParams hmmParams = new ToyHMMParams(2);
    Pair<List<Integer>, List<Integer>> generated = HMMUtils.generate(random, hmmParams, 3);
    List<Integer> observations = generated.getRight();
    
    // Here we can compute the exact log Z using sum product since we have a discrete HMM
    double exactLogZ = HMMUtils.exactDataLogProbability(hmmParams, observations);
    System.out.println("exact = " + Math.exp(exactLogZ));
    
    HMMProblemSpecification proposal = new HMMProblemSpecification(hmmParams, observations);
    SMCOptions options = new SMCOptions();
    ExhaustiveDebugRandom exhausiveRand = new ExhaustiveDebugRandom();
    options.random = exhausiveRand;
    options.nParticles = 2;
    options.essThreshold = 1.0;
    SMCAlgorithm<Integer> smc = new SMCAlgorithm<>(proposal, options);
    
    double expectation = 0.0;
    int nProgramTraces = 0;
    while (exhausiveRand.hasNext())
    {
      double logZ = smc.sample().logNormEstimate();
      expectation += Math.exp(logZ) * exhausiveRand.lastProbability();
      nProgramTraces++;
    }
    System.out.println("expectation = " + expectation); 
    System.out.println("nProgramTraces = " + nProgramTraces);
  }
}
