package simplesmc;

import java.util.ArrayList;
import java.util.Collections;
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
  
  int [][] ancestors;
  double [][] weights;
  
  public double sample(Random random)
  {
    ancestors  =  new int               [proposal.nIterations()][nParticles];
    weights =  new double            [proposal.nIterations()][nParticles];
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
    List<List<Integer>> result = _bs(nGens, nPart);
    for (List<Integer> item : result)
      Collections.reverse(item);
    return result;
  }
  
  private static List<List<Integer>> _bs(int nGens, int nPart)
  {
    List<List<Integer>> result = new ArrayList<List<Integer>>();
    if (nGens == 0)
    {
      result.add(new ArrayList<>());
      return result;
    }
    
    List<List<Integer>> prefixes = _bs(nGens - 1, nPart);
    for (List<Integer> prefix : prefixes)
    {
      int bound = prefix.isEmpty() ? nPart - 1 : prefix.get(prefix.size() - 1);
      for (int i = 0; i <= bound; i++)
      {
        List<Integer> item = new ArrayList<>(prefix);
        item.add(i);
        result.add(item);
      }
    }
    return result;
  }
  
  private static double permutNorm(int nGens, int nPart)
  {
    System.out.println("---");
    double outerSum = 0.0; 
    for (List<Integer> bs : bs(nGens, nPart))
    {
      double product = 1.0;
      
      for (int gen = 0; gen < nGens - 1; gen++)
      {
        int curB = bs.get(gen + 1);
        product *= 1.0 / (curB + 1);
      }
      product /= nPart;
      System.out.println("" + bs + "\t" + product);
      
      outerSum += product;
    }
    System.out.println("---");
    return outerSum;
  }
  
//  public double phi()
//  {
//    
//  }
  
  public double piExtended(double phi, double z)
  {
    int nGens = weights.length;
    int nPart = weights[0].length;
    double outerSum = 0.0; 
    for (List<Integer> bs : bs(nGens, nPart))
    {
      double product = weights[nGens - 1][bs.get(nGens - 1)];
      
      for (int gen = 0; gen < nGens - 1; gen++)
      {
        double innerSum = 0.0;
        int curB = bs.get(gen + 1);
        for (int part = 0; part <= curB; part++)
          innerSum += weights[gen][part];
        
        product *= innerSum / (curB + 1);
      }
      outerSum += product;
    }
    return outerSum * phi / z / nPart;
  }
  
  public double piExtended2(double phi, double z, List<Integer> bs)
  {
    int nGens = weights.length;
    double product = weights[nGens - 1][bs.get(nGens - 1)];
    
    for (int gen = 0; gen < nGens - 1; gen++)
    {
      double innerSum = 0.0;
      int curB = bs.get(gen + 1);
      for (int part = 0; part <= curB; part++)
        innerSum += weights[gen][part];
      
      product *= innerSum;
    }
    return product * phi / z;
  }
  
  static enum Z_Estimator
  {
    FULL_RB {
      @Override
      double compute(double[][] weights, int[][] ancestors, Random random)
      {
        int nGens = weights.length;
        int nPart = weights[0].length;
        
        double outerSum = 0.0;
        for (List<Integer> bs : bs(nGens, nPart))
        {
          double product = weights[nGens - 1][bs.get(nGens - 1)];
          
          for (int gen = 0; gen < nGens - 1; gen++)
          {
            double innerSum = 0.0;
            int curB = bs.get(gen + 1);
            for (int part = 0; part <= curB; part++)
              innerSum += weights[gen][part];
            product *= innerSum / (curB+1);
          }
          outerSum += product;
        }
        return outerSum / nPart;
      }
    },
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
    NEW_ESTIMATOR {

      @Override
      double compute(double[][] weights, int[][] ancestors, Random rand)
      {
        int nGens = weights.length;
        int nPart = weights[0].length;
        double denom = 0.0;
        double num = 0.0;
        for (List<Integer> bs : bs(nGens, nPart)) 
        {
          double product = 1.0;
          for (int t = 0; t < nGens; t++)
            product *= weights[t][bs.get(t)];
          num += product;
          denom++;
        }
        
        return num/denom;
      }
      
    },
    CONJECTURE_ARNAUD {

      @Override
      double compute(double[][] weights, int[][] ancestors, Random rand)
      {
        int nGens = weights.length;
        int nPart = weights[0].length;
        
        double outerSum = 0.0;
        
        for (List<Integer> bs : bs(nGens, nPart))
        {
          double product = 1.0;
          
          for (int t = 0; t < nGens; t++)
          {
            double innerSum = 0.0;
            
            for (int i = 0; i < bs.get(t); i++)
              innerSum += weights[t][i];
            
            product *= innerSum / (1+bs.get(t));
          }
          
          outerSum += product;
        }
        
        return outerSum / Math.pow(nPart, nGens);
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
    int nGens = 3;
    int nParts = 2;
    System.out.println("nGen = " + nGens);
    System.out.println("nParts = " + nParts);

    System.out.println("permutNorm = " + permutNorm(nGens, nParts));
    
    System.out.println(bs(nGens, nParts));
    
    // Create a synthetic dataset
    Random random = new Random(1);
    ToyHMMParams hmmParams = new ToyHMMParams(2); // 2 states HMM
    Pair<List<Integer>, List<Integer>> generated = HMMUtils.generate(random, hmmParams, nGens);
    List<Integer> observations = generated.getRight();
    System.out.println("Observations : " + observations);
    
    // Here we can compute the exact log Z using sum product since we have a discrete HMM
    double exactLogZ = HMMUtils.exactDataLogProbability(hmmParams, observations);
    double exactZ =  Math.exp(exactLogZ);
    System.out.println("exact = " + exactZ);
    
    HMMProblemSpecification proposal = new HMMProblemSpecification(hmmParams, observations);
    
//    List<Integer> fixedBs = bs(nGens, nParts).get(4);
    for (Z_Estimator estimator : Z_Estimator.values())
    {
      
      System.out.println("Estimator: " + estimator);
      ExhaustiveDebugRandom exhausiveRand = new ExhaustiveDebugRandom();
      NaiveSequentiallyInteracting<Integer> smc = new NaiveSequentiallyInteracting<>(proposal, nParts, estimator);
      
      double expectation = 0.0;
      int nProgramTraces = 0;
      double variance = 0.0;
//      double piNorm = 0.0;
//      double phiNorm = 0.0;
//      double piNorm2 = 0.0;
      while (exhausiveRand.hasNext())
      {
        double zHat = smc.sample(exhausiveRand);
//        phiNorm += exhausiveRand.lastProbability();
//        piNorm  += smc.piExtended (exhausiveRand.lastProbability(), exactZ);
//        piNorm2 += smc.piExtended2(exhausiveRand.lastProbability(), exactZ, fixedBs);
        expectation += zHat * exhausiveRand.lastProbability();
        variance += (zHat - exactZ) * (zHat - exactZ)* exhausiveRand.lastProbability();
        nProgramTraces++;
      }
      
//      System.out.println("\tphiNorm = " + phiNorm);
//      System.out.println("\tpiNorm = " + piNorm);
//      System.out.println("\tpiNorm2 = " + piNorm2);
      System.out.println("\tbias = " + (exactZ - expectation)); 
      System.out.println("\tstd_dev = " + Math.sqrt(variance));
      System.out.println("\tnProgramTraces = " + nProgramTraces);
    }
  }
}
