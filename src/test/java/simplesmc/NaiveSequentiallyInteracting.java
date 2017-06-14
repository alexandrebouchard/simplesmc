package simplesmc;

import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import bayonet.distributions.ExhaustiveDebugRandom;
import bayonet.distributions.Multinomial;
import bayonet.distributions.Random;
import simplesmc.hmm.HMMProblemSpecification;
import simplesmc.hmm.HMMUtils;
import simplesmc.hmm.ToyHMMParams;

public class NaiveSequentiallyInteracting<P>
{
  final ProblemSpecification<P> proposal;
  final int nParticles;
  
  public NaiveSequentiallyInteracting(ProblemSpecification<P> proposal, int nParticles)
  {
    super();
    this.proposal = proposal;
    this.nParticles = nParticles;
  }

  public double sample(Random random)
  {
    int [][] ancestors =   new int           [proposal.nIterations()][nParticles];
    double [][] weights =  new double        [proposal.nIterations()][nParticles];
    @SuppressWarnings("unchecked")
    P [][] particles =     (P[][]) new Object[proposal.nIterations()][nParticles];
    
    // initial particle
    {
      // initial generation
      {
        Pair<Double, P> proposeInitial = proposal.proposeInitial(random);
        weights  [0][0] = Math.exp(proposeInitial.getLeft());
        particles[0][0] = proposeInitial.getRight();
        ancestors[0][0] = -1;
      }
      
      // subsequent generations
      for (int gen = 1; gen < proposal.nIterations(); gen++)
      {
        Pair<Double, P> proposeNext = proposal.proposeNext(gen - 1, random, particles[gen - 1][0]);
        weights  [gen][0] = Math.exp(proposeNext.getLeft());
        particles[gen][0] = proposeNext.getRight();
        ancestors[gen][0] = 0;
      }
    }
    
    // subsequent particles
    for (int part = 1; part < nParticles; part++)
    {
      // initial generation
      {
        Pair<Double, P> proposeInitial = proposal.proposeInitial(random);
        weights  [0][part] = Math.exp(proposeInitial.getLeft());
        particles[0][part] = proposeInitial.getRight();
        ancestors[0][part] = -1;
      }
      
      // subsequent generations
      for (int gen = 1; gen < proposal.nIterations(); gen++)
      {
        // resampling
        double [] prevWeights = weights[gen - 1].clone();
        Multinomial.normalize(prevWeights);
        int sampledIndex = random.nextCategorical(prevWeights);
        // proposal
        Pair<Double, P> proposeNext = proposal.proposeNext(gen - 1, random, particles[gen - 1][sampledIndex]);
        weights[gen][part] = Math.exp(proposeNext.getLeft());
        particles[gen][part] = proposeNext.getRight();
        ancestors[gen][part] = sampledIndex;
      }
    }
    
    // compute Z estimator (naive version)
    
    double [] normalizedLastGenWeights = weights[proposal.nIterations() - 1].clone();
    Multinomial.normalize(normalizedLastGenWeights);
    int previousIndex = nParticles - 1;
    int currentIndex = random.nextCategorical(normalizedLastGenWeights);
    
    double product = 1.0;
    for (int t = proposal.nIterations() - 1; t >= 0; t--)
    {
      double sum = 0.0;
      for (int i = 0; i <= previousIndex; i++)
        sum += weights[t][i];
      product *= sum / (previousIndex + 1);
      
      previousIndex = currentIndex;
      currentIndex = ancestors[t][currentIndex];
    }
    
    return product;
  }
  
  public static void main(String [] args)
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
    
    ExhaustiveDebugRandom exhausiveRand = new ExhaustiveDebugRandom();
    NaiveSequentiallyInteracting<Integer> smc = new NaiveSequentiallyInteracting<>(proposal, 2);
    
    double expectation = 0.0;
    int nProgramTraces = 0;
    while (exhausiveRand.hasNext())
    {
      double z = smc.sample(exhausiveRand);
      expectation += z * exhausiveRand.lastProbability();
      nProgramTraces++;
    }
    System.out.println("expectation = " + expectation); 
    System.out.println("nProgramTraces = " + nProgramTraces);
  }
}
