package simplesmc;

import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.Assert;
import org.junit.Test;

import simplesmc.hmm.HMMProblemSpecification;
import simplesmc.hmm.HMMUtils;
import simplesmc.hmm.ToyHMMParams;
import tutorialj.Tutorial;



public class TestSMC
{
  /**
   * We show here a simple example where the target distribution is a finite HMM. This is 
   * only for test purpose, to ensure that the estimate of the log normalization is close to 
   * the true value.
   * 
   * Note the only step required to customize the SMC algorithm to more complex and interesting problem 
   * is to create a class that implements ``simplesmc.ProblemSpecification``.
   */
  @Tutorial(showLink = true, linkPrefix = "src/test/java/")
  @Test
  public void testSMC()
  {
    // Create a synthetic dataset
    Random random = new Random(1);
    ToyHMMParams hmmParams = new ToyHMMParams(5);
    Pair<List<Integer>, List<Integer>> generated = HMMUtils.generate(random, hmmParams, 10);
    List<Integer> observations = generated.getRight();
    
    // Here we can compute the exact log Z using sum product since we have a discrete HMM
    double exactLogZ = HMMUtils.exactDataLogProbability(hmmParams, observations);
    System.out.println("exact = " + exactLogZ);
    
    // Run SMC to ensure correctness of our implementation
    HMMProblemSpecification proposal = new HMMProblemSpecification(hmmParams, observations);
    SMCOptions options = new SMCOptions();
    options.nParticles = 1_000;
    SMCAlgorithm<Integer> smc = new SMCAlgorithm<>(proposal, options);
    
    // Check they agree within 1%
    double approxLogZ = smc.sample().logNormEstimate();
    System.out.println("estimate = " + approxLogZ);
    double tol = Math.abs(exactLogZ / 100.0);
    System.out.println("tol = " + tol);
    Assert.assertEquals(exactLogZ, approxLogZ, tol);
  }
}
