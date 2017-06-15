package simplesmc;

import java.util.ArrayList;
import java.util.Iterator;
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
  final Z_Estimator estimator;

  public NaiveSequentiallyInteracting(ProblemSpecification<P> proposal, int nParticles, Z_Estimator estimator)
  {
    this.proposal = proposal;
    this.nParticles = nParticles;
    this.estimator = estimator;
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
    return estimator.compute(weights, ancestors, random);
  }
  
  private static List<List<Integer>> bs(int nGens, int nPart)
  {
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    if (nGens == 0)
    {
      result.add(new ArrayList<>());
      return result;
    }
    
    List<List<Integer>> prefixes = bs(nGens - 1, nPart);
    for (List<Integer> prefix : prefixes)
      for (int i = 0; i < nPart; i++)
      {
        List<Integer> item = new ArrayList<>(prefix);
        item.add(i);
        result.add(item);
      }
    return result;
  }
  
  static enum Z_Estimator
  {
    SIMPLE {
      @Override
      double compute(double[][] weights, int[][] ancestors, Random random)
      {
        int nGens = weights.length;
        int nPart = weights[0].length;
        double [] normalizedLastGenWeights = weights[nGens - 1].clone();
        Multinomial.normalize(normalizedLastGenWeights);
        int previousIndex = nPart - 1;
        int currentIndex = random.nextCategorical(normalizedLastGenWeights);
        
        double product = 1.0;
        for (int t = nGens - 1; t >= 0; t--)
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
    },
    SIMPLE_RB {
      @Override
      double compute(double[][] weights, int[][] ancestors, Random random)
      {
        int nGens = weights.length;
        int nPart = weights[0].length;
        double [] normalizedLastGenWeights = weights[nGens - 1].clone();
        Multinomial.normalize(normalizedLastGenWeights);
        
        double outerSum = 0.0;
        
        for (int j = 0; j < nPart; j++)
        {
          int previousIndex = nPart - 1;
          int currentIndex = j;
          
          double product = 1.0;
          for (int t = nGens - 1; t >= 0; t--)
          {
            double sum = 0.0;
            for (int i = 0; i <= previousIndex; i++)
              sum += weights[t][i];
            product *= sum / (previousIndex + 1);
            
            previousIndex = currentIndex;
            currentIndex = ancestors[t][currentIndex];
          }
          
          outerSum += product * normalizedLastGenWeights[j];
        }
        return outerSum;
      }
    },
    FULL_RB {

      @Override
      double compute(double[][] weights, int[][] ancestors, Random random)
      {
        int nGens = weights.length;
        int nPart = weights[0].length;
        
        double [][] normWs = new double[nGens][];
        for (int gen = 0; gen < nGens; gen++)
        {
          normWs[gen] = weights[gen].clone();
          Multinomial.normalize(normWs[gen]);
        }
        
        double outerSum = 0.0;
        for (List<Integer> bs : bs(nGens, nPart))
        {
          Iterator<Integer> bIter = bs.iterator();

          int previousIndex = nPart - 1;
          int currentIndex = bIter.next();
          
          double product = 1.0;
          for (int t = nGens - 1; t >= 0; t--)
          {
            System.out.println(nGens - t - 1);
            product *= normWs[t][bs.get(nGens - t - 1)];
            
            double sum = 0.0;
            for (int i = 0; i <= previousIndex; i++)
              sum += weights[t][i];
            product *= sum / (previousIndex + 1);
            
            previousIndex = currentIndex;
            try { currentIndex = bIter.next(); } catch (Exception e) {}
          }
          outerSum += product;
        }
        
        return outerSum;
      }

      
    },
    NAIVE {
      @Override
      double compute(double[][] weights, int[][] ancestors, Random rand)
      {
        int nGens = weights.length;
        int nPart = weights[0].length;
        double product = 1.0;
        
        for (int gen = 0; gen < nGens; gen++)
        {
          double sum = 0.0;
          
          for (int i = 0; i < nPart; i++)
            sum += weights[gen][i];
          
          product *= sum / nPart;
        }
        
        return product;
      }
    };
    
    abstract double compute(double[][] weights, int[][] ancestors, Random rand);
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
    
    for (List<Integer> bs : bs(3, 2))
    {
      System.out.println(bs);
    }
    
    for (Z_Estimator estimator : Z_Estimator.values())
    {
      System.out.println("Estimator: " + estimator);
      ExhaustiveDebugRandom exhausiveRand = new ExhaustiveDebugRandom();
      NaiveSequentiallyInteracting<Integer> smc = new NaiveSequentiallyInteracting<>(proposal, 3, estimator);
      
      double expectation = 0.0;
      int nProgramTraces = 0;
      double variance = 0.0;
      while (exhausiveRand.hasNext())
      {
        double z = smc.sample(exhausiveRand);
        expectation += z * exhausiveRand.lastProbability();
        variance += (z - Math.exp(exactLogZ)) * (z - Math.exp(exactLogZ))* exhausiveRand.lastProbability();
        nProgramTraces++;
      }
      System.out.println("\texpectation = " + expectation); 
      System.out.println("\tstd_dev = " + Math.sqrt(variance));
      System.out.println("\tnProgramTraces = " + nProgramTraces);
    }
  }
}
