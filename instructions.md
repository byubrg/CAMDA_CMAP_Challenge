## Getting Started

1. Please create your own forked repository of WishBuilder by clicking on "Fork" (upper right hand corner of the webpage). Please make all of your future pull requests from your forked repository.

2. Please find your repository api by navigating to your new forked repository and clicking on "clone or download". Your repository api should look something like this:

  ```
  git@github.com:glenrs/WishBuilder.git
  ```

3. You will then need to clone your forked repository on your computer.

  ```
  git clone <your-forked-repository-api> 
  ```

  ```
  git clone git@github.com:glenrs/WishBuilder.git
  ```

4. Navigate into the directory (cd CAMDA_CMAP_Challenge). You are now on a cloned version on your forked repository. The changes you make are not backed up on GitHub until you push them to GitHub.

## Working on a Branch

1. You will never edit your master branch of your forked repository. You can always update your forked master branch with the following code.

  ```
  git pull git@github.com:byubrg/CAMDA_CMAP_Challenge.git master 
  ```

2. Before making new branches you will want to make sure you get the latest version of the master branch so that you can see the most recent changes. You are now ready to create a branch.

  ```
  git checkout -b <name-of-branch> 
  ```

3. On this branch you can make the necessary changes so that you can merge your changes with the BRG/CAMDA_CMAP_Challenge repository. After making changes you will want to save your changes locally by committing to your local branch so that you can switch to other branches. But before doing so you will need to make some adjustments.

  ```
  git add --all 
  ```

4. Before commiting you can check to see if you added the information that you wanted to add.

  ```
  git status
  ```

5. If there is something that you do not want commited you wan reset your temporary storage and then repeat 3.

  ```
  git reset
  ```

6. You are now ready to commit. When you commit it is convenient to make a message so that you remember what you were commiting. This will help you in the future so that you can come back to possible previous changes.

  ```
  git commit -m "<This is your message>"
  ```

7. You have now stored your information on your local branches. You can now switch between branches safely. You are now prepared to push to GitHub. You will do this by pushing to your branch on your forked repository.

  ```
  git push origin <name-of-branch>
  ```

8. To add this to the master repository you will now need to navigate to the repository for the group under [Pull Requests](https://github.com/byubrg/CAMDA_CMAP_Challenge/pulls). You can now create your pull request by clicking on "New Pull Request". After click on "compare across forks" and navigate to the fork you would like to merge with the master branch on the club site. Next click on "Create Pull Request". On the next page you will see the commits on your branch and you can navigate to other windows that will allow you to see the changes that you made. If any of the changes look unfamiliar close your pull request. If it looks good, message me and I will merge your pull request after making sure there are no conflicts.
